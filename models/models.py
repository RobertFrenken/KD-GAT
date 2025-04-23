import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv, JumpingKnowledge
class GATBinaryClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, out_channels):
        super(GATBinaryClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1)
        self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1)
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
class GATWithJK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # GAT layers
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(
                GATConv(in_dim, hidden_channels, heads=heads, concat=True)
            )
        
        # JK aggregation (LSTM mode)
        self.jk = JumpingKnowledge(
            mode="lstm",
            channels=hidden_channels * heads,
            num_layers=num_layers
        )
        
        # Final projection
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            xs.append(x)
        
        # Aggregate layer outputs
        x = self.jk(xs)
        x = global_mean_pool(x, batch)  # Readout layer
        x = self.lin(x)
        return x

class GATWithSkips(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_heads=8):
        super().__init__()
        # Layer 1: Initial GAT layer
        self.gat1 = GATConv(in_channels, 64, heads=num_heads)
        self.skip_proj1 = nn.Linear(in_channels, 64 * num_heads)  # For dimension matching
        
        # Layer 2: Intermediate GAT layer
        self.gat2 = GATConv(64 * num_heads, 128, heads=num_heads//2)
        self.skip_proj2 = nn.Linear(64 * num_heads, 128 * (num_heads//2))
        
        # Layer 3: Final GAT layer with skip integration
        self.gat3 = GATConv(128 * (num_heads//2) + 64 * num_heads, num_classes, heads=1)  # Concatenated input

    def forward(self, x, edge_index):
        # Layer 1
        x1 = self.gat1(x, edge_index)
        x_skip1 = self.skip_proj1(x)  # Project original features
        
        # Layer 2 with skip from Layer 1
        x2 = self.gat2(x1, edge_index)
        x_skip2 = self.skip_proj2(x1)  # Project Layer 1 output
        
        # Layer 3 with concatenated skips
        x3 = torch.cat([x_skip1, x_skip2], dim=1)
        return self.gat3(x3, edge_index)

class EnhancedGAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gat1 = GATConv(in_dim, 256, heads=4)
        self.gat2 = GATConv(256*4, 256, heads=2)
        self.gat3 = GATConv(256*2 + 256*4, out_dim, heads=1)
        
        self.skip_proj = nn.Sequential(
            nn.Linear(in_dim, 256*4),
            nn.ELU()
        )

    def forward(self, x, edge_index):
        identity = self.skip_proj(x)
        
        x1 = F.elu(self.gat1(x, edge_index))
        x2 = F.elu(self.gat2(x1, edge_index))
        
        combined = torch.cat([identity, x2], dim=1)
        return self.gat3(combined, edge_index)


if __name__ == '__main__':
    net = GATWithJK(10, 8, 1)

    def model_characteristics(model):
        num_params = sum(p.numel() for p in model.parameters())
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f'Number of Parameters: {num_params:.3f}')
        print(f'Model size: {size_all_mb:.3f} MB')

    model_characteristics(net)
    print(net)