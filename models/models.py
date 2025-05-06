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
# Modify the GATWithJK model to include dropout
class GATWithJK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=3, heads=4, dropout=0.2, num_fc_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        
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

        # Fully connected layers after pooling
        self.fc_layers = torch.nn.ModuleList()
        fc_input_dim = hidden_channels * heads
        for _ in range(num_fc_layers - 1):
            self.fc_layers.append(torch.nn.Linear(fc_input_dim, fc_input_dim))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.Dropout(p=dropout))
        self.fc_layers.append(torch.nn.Linear(fc_input_dim, out_channels))  # Final output layer
        
        # Final projection
        # self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)
        

    def forward(self, data, return_intermediate=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)  # Add dropout
            xs.append(x)
        
        if return_intermediate:
            return xs
        
        # Aggregate layer outputs
        x = self.jk(xs)
        x = global_mean_pool(x, batch)  # Readout layer
        # Pass through fully connected layers + final output layer
        for layer in self.fc_layers:
            x = layer(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)  # Add dropout before final layer
        # x = self.lin(x)
        return x


if __name__ == '__main__':
    # Knowledge Distillation Scenario
    teacher_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=5, heads=8)
    student_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=2, heads=4)
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

    model_characteristics(teacher_model)
    print(teacher_model)
    model_characteristics(student_model)
    print(student_model)