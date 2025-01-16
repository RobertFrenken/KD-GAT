import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.classifier = nn.Linear(output_dim * 2, 1)  # Adjust the linear layer for edge embeddings

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        x = self.classifier(edge_embeddings)  # Apply the classifier layer to edge embeddings
        return torch.sigmoid(x)  # Use sigmoid activation for binary classification