# 文件1：gat_with_focal_loss.py

import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from models.models import GATWithJK
from preprocessing import graph_creation
from torch.nn.functional import binary_cross_entropy_with_logits

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# ----------
# 参数设置
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_FOLDER = r"D:\\quadeer\\task1\\can-train-and-test-v1.5\\set_01"

# ----------
# 数据加载

dataset = graph_creation(ROOT_FOLDER)
subset_size = int(0.2 * len(dataset))
subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
subset = torch.utils.data.Subset(dataset, subset_indices)
train_size = int(0.8 * len(subset))
test_size = len(subset) - train_size
train_dataset, test_dataset = random_split(subset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------
# 模型与训练
model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=5, heads=8).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = FocalLoss(alpha=1, gamma=2)

model.train()
for epoch in range(20):
    total_loss = 0
    for data in train_loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data).view(-1)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ----------
# 测试
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(DEVICE)
        out = model(data).view(-1)
        pred = (torch.sigmoid(out) > 0.5).int().cpu()
        y_true.append(data.y.cpu())
        y_pred.append(pred)

y_true = torch.cat(y_true).numpy()
y_pred = torch.cat(y_pred).numpy()

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))


# # 文件2：gat_with_focal_attention.py

# # 同上，只是模型定义加权 attention（示意）

# from torch_geometric.nn import GATConv
# class GATWithAttentionMask(GATWithJK):
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             if self.training:
#                 if hasattr(data, 'y'):
#                     mask = (data.y == 1).float().view(-1, 1)
#                     x = x * (1 + 0.5 * mask)  # 攻击类加权
#             x = torch.relu(x)
#         x = self.jump(x)
#         x = self.lin(x)
#         return x.view(-1)

# # 其余与第一份代码相同，替换模型为 GATWithAttentionMask 即可