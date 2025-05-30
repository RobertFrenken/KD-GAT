import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import classification_report
from preprocessing import graph_creation
import torch.nn as nn
class GATWithAttentionMask(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1))
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            if hasattr(data, 'y'):
                mask = (data.y[data.batch] == 1).float().unsqueeze(1)
                x = x * (1 + 0.5 * mask)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x.view(-1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:\robert\CAN-Graph\saved_models\best_set01_gat_attention_alpha0995_gamma15.pth"
ROOT_FOLDER = r"D:\quadeer\task1\can-train-and-test-v1.5\set_01"


from models.models import GATWithJK

model = GATWithJK(
    in_channels=10,
    hidden_channels=32,
    out_channels=1,
    num_layers=5,
    heads=8
).to(DEVICE)



model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# load new data to test
print("Loading new subset of dataset …")
dataset = graph_creation(ROOT_FOLDER)
subset_size = int(0.2 * len(dataset))
subset_idx = np.random.choice(len(dataset), subset_size, replace=False)
subset = Subset(dataset, subset_idx)
test_loader = DataLoader(subset, batch_size=32, shuffle=False)

print("Evaluating on new 20% sample …")
y_true, y_pred = [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(DEVICE)
        logits = model(data)
        preds = (torch.sigmoid(logits) > 0.5).int().cpu()
        y_true.append(data.y.cpu())
        y_pred.append(preds)

y_true = torch.cat(y_true).numpy()
y_pred = torch.cat(y_pred).numpy()
print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
