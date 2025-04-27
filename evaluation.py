import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
from models.models import GATWithJK
from preprocessing import graph_creation
from training_utils import PyTorchTrainer, PyTorchDistillationTrainer, DistillationTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, data_loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            outputs = model(data)
            preds = (outputs > 0.5).float()  # Assuming binary classification
            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    # Flatten the lists
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return accuracy, precision, recall, f1


def main():
    x =1 # Placeholder for the main function logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=5, heads=8).to(device)
    student_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=2, heads=4).to(device)

    test_datasets = [
    {"folder": "datasets/can-train-and-test-v1.5/set_01", 
    "teacher_weight": "saved_models/best_teacher_model_set_01.pth", 
    "student_weight": "saved_models/final_student_model_set_01.pth"},

    {"folder": "datasets/can-train-and-test-v1.5/set_02", 
    "teacher_weight": "saved_models/best_teacher_model_set_02.pth", 
    "student_weight": "saved_models/final_student_model_set_02.pth"},

    {"folder": "datasets/can-train-and-test-v1.5/set_03", 
    "teacher_weight": "saved_models/best_teacher_model_set_03.pth", 
    "student_weight": "saved_models/final_student_model_set_03.pth"},

    {"folder": "datasets/can-train-and-test-v1.5/set_04", 
    "teacher_weight": "saved_models/best_teacher_model_set_04.pth", 
    "student_weight": "saved_models/final_student_model_set_04.pth"},

    {"folder": "datasets/can-train-and-test-v1.5/hcrl-ch", 
    "teacher_weight": "saved_models/best_teacher_model_ch.pth", 
    "student_weight": "saved_models/final_student_model_ch.pth"},
     
    {"folder": "datasets/can-train-and-test-v1.5/hcrl-sa", 
    "teacher_weight": "saved_models/best_teacher_model_hcrl_sa.pth", 
    "student_weight": "saved_models/final_student_model_hcrl_sa.pth"},
    ]

    # Iterate through each dataset
    sample_dataset = [    {"folder": "datasets/can-train-and-test-v1.5/hcrl-ch", 
    "teacher_weight": "saved_models/best_teacher_model_ch.pth", 
    "student_weight": "saved_models/final_student_model_ch.pth"},]


    for dataset_info in sample_dataset:
        root_folder = dataset_info["folder"]
        teacher_weight = dataset_info["teacher_weight"]
        student_weight = dataset_info["student_weight"]

        print(f"Evaluating dataset in root folder: {root_folder}")

        # Load the test dataset using graph_creation
        test_dataset = graph_creation(root_folder, folder_type="test_")
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Load the teacher model
        teacher_model.load_state_dict(torch.load(teacher_weight))
        teacher_model.eval()

        # Load the student model
        student_model.load_state_dict(torch.load(student_weight))
        student_model.eval()

        # Evaluate the teacher model
        teacher_metrics = evaluate_model(teacher_model, test_loader, device)
        print(f"Teacher Model - Accuracy: {teacher_metrics[0]:.4f}, Precision: {teacher_metrics[1]:.4f}, Recall: {teacher_metrics[2]:.4f}, F1 Score: {teacher_metrics[3]:.4f}")

        # Evaluate the student model
        student_metrics = evaluate_model(student_model, test_loader, device)
        print(f"Student Model - Accuracy: {student_metrics[0]:.4f}, Precision: {student_metrics[1]:.4f}, Recall: {student_metrics[2]:.4f}, F1 Score: {student_metrics[3]:.4f}")

        print("-" * 50)

        
       
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")