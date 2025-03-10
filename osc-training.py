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
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import sys
import time

def dataset_creation(path):
    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'CAN ID','DLC','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8', 'label'] 

    df['Node'] = df['CAN ID']
    df['Edge'] = df['CAN ID'].shift(-1)

    df = df.apply(lambda x: x.apply(hex_to_decimal))

    # Drop the last row
    df = df.drop(df.index[-1])

    # Drop rows where 'Data3' contains 'R'
    # I need to make this more comprehensive
    df_dropped = df[df['Data3'] != 'R']
    df= df_dropped

    # reencode the labels
    df['label'] = df['label'].replace({'R': 0, 'T': 1})

    # Replace NaN values with zero
    df = df.fillna(0)

    arr = df[['Node', 'Edge', 'label']].to_numpy(dtype=float)

    return arr

def hex_to_decimal(x):
        if x is None or x == 'None':
            return None
        try:
            return int(x, 16)
        except (ValueError, TypeError):
            return x

def create_graphs(data, window_size, stride):
    graphs = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i+window_size]
        graph = create_single_graph(window)
        graphs.append(graph)
    return graphs

def create_single_graph(window_data):
        x = torch.tensor(window_data[:, 0:1], dtype=torch.float)
        
        edge_index = _get_edge_index(window_data) # call the edge index function here
        
        label = window_data[:, -1] # last column are the labels
        y = torch.tensor([1 if np.any(label == 1) else 0], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

def _get_edge_index(window_data: np.ndarray) -> torch.Tensor:
        num_nodes = window_data.shape[0]
        edge_index = torch.tensor([np.arange(num_nodes - 1), np.arange(1, num_nodes)], dtype=torch.long)
        return edge_index

class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def train(model, optimizer, train_loader, device):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()  # Squeeze the output to match the target shape
        loss = F.binary_cross_entropy_with_logits(out, data.y.float())
        loss.backward()
        optimizer.step()

def test(loader, model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()  # Squeeze the output to match the target shape
            # print(out)
            pred = (out > 0).long()
            correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # GAT(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Global mean pooling
        x = self.fc(x)
        return x

def main():
    # .. goes up one level in directory
    # path = r'../datasets/Car-Hacking Dataset/Fuzzy_dataset.csv'
    # path = r'../datasets/Car-Hacking Dataset/DoS_dataset.csv'
    # path = r'../datasets/Car-Hacking Dataset/gear_dataset.csv'
    path = r'datasets/Car-Hacking Dataset/RPM_dataset.csv'
    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'CAN ID','DLC','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8', 'label'] 
    
    arr = dataset_creation(path)
    list_graphs = create_graphs(arr, window_size=50, stride=50)
    # Create the dataset
    dataset = GraphDataset(list_graphs)

    num_epochs = 10

    train_ratio = 0.8

    # Calculate the number of samples for training and testing
    # Split the dataset into training and test sets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # model = AnomalyGNN(num_node_features=50, hidden_channels=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'Model is using device: {device}')
    model = GNN(in_channels=1, hidden_channels=16, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch).squeeze() 
            # print(out)
            # print(batch.y)
            loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
            # loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    train_acc = test(train_loader, model, device)
    test_acc = test(test_loader, model, device)
    print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")