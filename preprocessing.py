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
