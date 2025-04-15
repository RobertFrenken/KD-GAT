import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from models.models import GATBinaryClassifier, GATWithJK
from preprocessing import GraphDataset, graph_creation
from training_utils import PyTorchTrainer, PyTorchDistillationTrainer, DistillationTrainer, TrainingOrchestrator
WANDB_API_KEY = "02399594f766bc76e4af844217bf8188630cae40"
@hydra.main(config_path="conf", config_name="base", version_base=None)

def main(config: DictConfig):

    # add hydra instantiation to the script

    config_dict = OmegaConf.to_container(config, resolve=True)
    print(config_dict)

    # make device the first line
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Model is using device: {device}')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    path = r'datasets/Car-Hacking Dataset/Fuzzy_dataset.csv'
    path = r'datasets/Car-Hacking Dataset/DoS_dataset.csv'
    path = r'datasets/Car-Hacking Dataset/gear_dataset.csv'
    path = r'datasets/Car-Hacking Dataset/RPM_dataset.csv'
    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'CAN ID','DLC','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8', 'label']
    combined = config_dict['combined']
    DATASIZE = config_dict['datasize']

    dataset = graph_creation(combined, path, window_size=50, stride=50)

    # hyperparameters from yaml file
    EPOCHS = config_dict['epochs']
    LR = config_dict['lr']
    BATCH_SIZE = config_dict['batch_size']
    TRAIN_RATIO = config_dict['train_ratio']

    print("Size of the total dataset: ", len(dataset))
   

    # Calculate the number of samples for training and testing
    # Split the dataset into training and test sets
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)
    print('Size of DATASIZE: ', DATASIZE)
    print('Size of Training dataset: ', len(train_dataset))
    print('Size of Testing dataset: ', len(test_dataset))
    

    if DATASIZE < 1.0:
        subset_size = int(len(train_dataset) * DATASIZE)  # Fraction of the total dataset
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        subset = Subset(train_dataset, indices)
        train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Size of Training dataloader: ', len(train_loader))
    print('Size of Testing dataloader: ', len(test_loader))
    print('Size of Training dataloader (samples): ', len(train_loader.dataset))
    print('Size of Testing dataloader (samples): ', len(test_loader.dataset))

    

    # model = GATBinaryClassifier(in_channels=1, hidden_channels=32, num_heads=16, out_channels=1).to(device)
    # default 3 layers and 4 heads
    model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    model_path = 'best_model.pth'

    start_time = time.time()
    # trying with the PyTorchTrainer class
    # trainer = PyTorchTrainer(model, optimizer, criterion, device)
    # # Training loop
    # for epoch in range(EPOCHS):
    #     trainer.train_one_epoch(train_loader)
    #     trainer.validate(test_loader)
        
    #     # Get metrics
    #     metrics = trainer.report_latest_metrics()
    #     print(f"Epoch {epoch+1}:")
    #     print(f"  Train Loss: {metrics['train']['loss']:.4f}")
    #     print(f"  Val Accuracy: {metrics['val']['accuracy']:.2f}")


    # Knowledge Distillation Scenario
    # Initialize models
    teacher_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=5, heads=8).to(device)
    student_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=2, heads=4).to(device)

    # Initialize the DistillationTrainer
    trainer = DistillationTrainer(
        teacher=teacher_model,
        student=student_model,
        device=device,
        teacher_epochs=EPOCHS,  # Train teacher for 50 epochs
        student_epochs=EPOCHS,  # Train student for 50 epochs
        distill_alpha=0.5,      # Weight for distillation loss
        warmup_epochs=10        # Warmup epochs for student training
    )

    # Train teacher first, then student
    print("Starting sequential training...")
    trainer.train_sequential(train_loader)

    # Save the final student model
    torch.save(student_model.state_dict(), 'final_student_model.pth')
    print("Final student model saved as 'final_student_model.pth'.")

    # Training loop
    # for epoch in range(EPOCHS):
    #     trainer.train_one_epoch(train_loader)
    #     trainer.validate(test_loader)
        
    #     metrics = trainer.report_latest_metrics()
    #     if epoch < trainer.warmup_epochs:
    #         print(f"Warmup Epoch {epoch+1}: Loss={metrics['train']['loss']:.4f}")
    #     else:
    #         print(f"Distillation Epoch {epoch+1}:")
    #         print(f"  Student Loss: {metrics['train']['student_loss']:.4f}")
    #         print(f"  Distill Loss: {metrics['train']['distill_loss']:.4f}")


    # # training loop. Function in training_utils.py
    # # training(EPOCHS, model, optimizer, criterion, train_loader, test_loader, device, model_path)
    
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Model Training Runtime: {elapsed_time:.4f} seconds")
    # # train_acc, train_f1 = evaluation(train_loader, model, device)
    # # test_acc, test_f1 = evaluation(test_loader, model, device)
    # # print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    # # Access best model
    # model.load_state_dict(trainer.best_model_weights)
    # torch.save(model.state_dict(), 'finaland_best_model.pth')
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")