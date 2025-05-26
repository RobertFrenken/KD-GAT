import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from models.models import GATWithJK
from preprocessing import graph_creation
from torch_geometric.nn import global_mean_pool
from sklearn.svm import SVC
from sklearn.metrics import classification_report
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_FOLDER = r"D:\quadeer\task1\can-train-and-test-v1.5\set_01"
MODEL_PATH = "saved_models/best_teacher_model_set_01.pth"

# load
dataset = graph_creation(ROOT_FOLDER)
subset_size = int(0.2 * len(dataset))#get total of 0.2 to test
subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
subset = torch.utils.data.Subset(dataset, subset_indices)

# 80 to train 20 to test
train_size = int(0.8 * len(subset))
test_size = len(subset) - train_size
train_dataset, test_dataset = random_split(subset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# load model
teacher = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=5, heads=8).to(DEVICE)
teacher.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

teacher.eval()

def extract_embeddings(model, loader):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv in model.convs:
                x = conv(x, edge_index)
                x = torch.relu(x)
            pooled = global_mean_pool(x, batch)
            embeddings.append(pooled.cpu())
            labels.append(data.y.cpu())
    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()


X_train, y_train = extract_embeddings(teacher, train_loader)
X_test, y_test = extract_embeddings(teacher, test_loader)

#train
svm = SVC(kernel='rbf', class_weight='balanced')  # 平衡类别权重应对不平衡问题
svm.fit(X_train, y_train)

# test svm
y_pred = svm.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"])
print(report)

import joblib
import os

save_path = r"D:\robert\CAN-Graph\saved_models\best_teacher_model_set_01_svm.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

joblib.dump(svm, save_path)
print(f"saved to: {save_path}")
