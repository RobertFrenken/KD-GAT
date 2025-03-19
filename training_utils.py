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
def train(model, optimizer, train_loader, device):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()  # Squeeze the output to match the target shape
        loss = F.binary_cross_entropy_with_logits(out, data.y.float())
        loss.backward()
        optimizer.step()

def evaluation(loader, model, device):
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

def training(EPOCHS, model, optimizer, criterion, train_loader, test_loader, device, model_path):

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            batch.to(device) # put batch tensor on the correct device
            out = model(batch).squeeze() 
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                data.to(device)
                outputs = model(data).squeeze() 
                loss = criterion(outputs, data.y.float())
                val_loss += loss.item()

        
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        val_loss /= len(test_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved with validation loss: {best_val_loss}')