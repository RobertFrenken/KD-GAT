import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from models.models import GNN, GATBinaryClassifier
from preprocessing import dataset_creation, create_graphs, GraphDataset, graph_creation
from training_utils import train, test
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
    combined = config['combined']

    dataset = graph_creation(combined, path, window_size=50, stride=50)

    # hyperparameters from yaml file
    EPOCHS = config_dict['epochs']
    LR = config_dict['lr']
    BATCH_SIZE = config_dict['batch_size']

    train_ratio = 0.8

    # Calculate the number of samples for training and testing
    # Split the dataset into training and test sets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    # model = GNN(in_channels=1, hidden_channels=16, out_channels=1).to(device)
    model = GATBinaryClassifier(in_channels=1, hidden_channels=32, num_heads=16, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    model_path = 'best_model.pth'

    start_time = time.time()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            batch.to(device) # put batch tensor on the correct device
            out = model(batch).squeeze() 
            # print(out)
            # print(batch.y)
            loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
            # loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                data.to(device)
                outputs = model(data).squeeze() 
                loss = F.binary_cross_entropy_with_logits(outputs, data.y.float())
                val_loss += loss.item()

        
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        val_loss /= len(test_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved with validation loss: {best_val_loss}')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model Training Runtime: {elapsed_time:.4f} seconds")
    train_acc = test(train_loader, model, device)
    test_acc = test(test_loader, model, device)
    print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

    torch.save(model.state_dict(), 'final_model.pth')
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")