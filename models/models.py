import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
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

class SkipProjection(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.proj = nn.Linear(in_feats, out_feats)
    
    def forward(self, x_skip):
        return self.proj(x_skip)

# Example: Adjust layer 1's output (64 features) to match layer 3 (128 features)
skip_adjust = SkipProjection(64, 128)
adjusted_skip = skip_adjust(layer1_output)
# For mismatched heads (e.g., layer1: 4 heads, layer3: 6 heads)
combined_heads = torch.cat([layer1_output, layer3_output], dim=-1)
combined_heads = nn.Linear(combined_heads.size(-1), target_dim)(combined_heads)
class GATResBlock(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super().__init__()
        self.gat = GATConv(in_feats, out_feats, num_heads)
        self.skip = nn.Linear(in_feats, out_feats * num_heads) if in_feats != out_feats else nn.Identity()
    
    def forward(self, x, edge_index):
        x_gat = self.gat(x, edge_index)
        x_skip = self.skip(x)
        return F.elu(x_gat + x_skip)


def forward(self, x, edge_index):
    x1 = self.layer1(x, edge_index)
    print(f"Layer 1 output shape: {x1.shape}")  # Debug dimensions
    x2 = self.layer2(x1, edge_index)
    x3 = self.layer3(x2 + x1_adjusted, edge_index)  # Adjusted skip
    return x3
skip_combined = torch.cat([x_main, adjusted_skip], dim=-1)
x_out = nn.Linear(2 * target_dim, target_dim)(skip_combined)


class GATBinaryClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, out_channels):
        super(GATBinaryClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index) # + x  # Skip connection
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)  # Readout layer
        x = self.fc(x)
        return x  # Classification layer
from torch_geometric.nn import GATConv, Linear, to_hetero

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


model = GAT(hidden_channels=64, out_channels=dataset.num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')


if __name__ == '__main__':
    # Initialize the model
    input_features = dataset.num_node_features
    num_classes = dataset.num_classes
    model = SimpleGNN(input_features, 16, num_classes)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Assuming you have a dataset loaded
    data = dataset[0]

    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    
    # LSTM
    # Example usage:
    input_size = 10  # number of features
    hidden_size = 20  # number of features in hidden state
    num_layers = 2  # number of stacked lstm layers
    output_size = 1  # number of output classes

    # Create the model
    model = SimpleLSTM(input_size, hidden_size, num_layers, output_size)

    # Example input (batch_size, sequence_length, input_size)
    x = torch.randn(32, 5, input_size)
    # Forward pass
    output = model(x)
    print(output.shape)  # Should be (32, 1)