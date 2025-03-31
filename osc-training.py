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
from training_utils import evaluation, training
WANDB_API_KEY = "02399594f766bc76e4af844217bf8188630cae40"
@hydra.main(config_path="conf", config_name="base", version_base=None)

def main(config: DictConfig):

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
   

    # Calculate the number of samples for training and testing
    # Split the dataset into training and test sets
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)
    # Generate random indices for the subset
    indices = np.random.choice(train_size, int(train_size*DATASIZE), replace=False)
    # Create the subset using the indices
    subset = Subset(train_dataset, indices)

    if DATASIZE < 1.0:
        train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Size of dataloader: ', len(train_loader))

    # model = GATBinaryClassifier(in_channels=1, hidden_channels=32, num_heads=16, out_channels=1).to(device)
    # default 3 layers and 4 heads
    model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    model_path = 'best_model.pth'

    start_time = time.time()

    # training loop. Function in training_utils.py
    training(EPOCHS, model, optimizer, criterion, train_loader, test_loader, device, model_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model Training Runtime: {elapsed_time:.4f} seconds")
    train_acc = evaluation(train_loader, model, device)
    test_acc = evaluation(test_loader, model, device)
    print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

    torch.save(model.state_dict(), 'final_model.pth')
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")