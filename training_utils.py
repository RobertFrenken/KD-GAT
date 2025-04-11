import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import graph_creation, GraphDataset
from models.models import GATWithJK

def evaluation(loader, model, device, desc="[Model]"):
    """
    Determines the accuracy of a model on a given dataset using a DataLoader.
    
    Args:
        loader (torch_geometric.loader.DataLoader): DataLoader for the dataset.
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device to run the model on.
        desc (str): Description of the model for logging.
        
    Returns:
        float: The accuracy of the model on the dataset.
        float: The F1 score of the model on the dataset.
    """
    model.eval()
    all_preds, all_labels = [], []
    start = time.time()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()  # Squeeze the output to match the target shape
            # print(out)
            pred = (out > 0).long()
            correct += (pred == data.y).sum().item()
            all_preds.append(pred.item())
            all_labels.append(data.y.item())
    
    end = time.time()
    total_time = end - start
    num_samples = len(loader.dataset)
    avg_time_ms = (total_time / num_samples) * 1000
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='binary')
    print(f"{desc} Inference Time => Total: {total_time:.2f}s | Avg/sample: {avg_time_ms:.2f} ms")
    
    return acc, f1

def training(EPOCHS, model, optimizer, criterion, train_loader, test_loader, device, model_path):
    """
    Determines the accuracy of a model on a given dataset using a DataLoader.
    
    Args:
        loader (torch_geometric.loader.DataLoader): DataLoader for the dataset.
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device to run the model on.
        
    Returns:
        float: The accuracy of the model on the dataset.
    """
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch.to(device) # put batch tensor on the correct device
            out = model(batch).squeeze()
            #print("Output shape: ", out.shape)
            # print("Batch shape: ", batch.y.shape)  
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

        val_loss /= len(test_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved with validation loss: {best_val_loss}')


def distill_loss_fn(student_logits, teacher_logits, labels, alpha=0.5, T=2.0):
    """
    Calculates the distillation loss using both soft and hard labels.
    
    Args:
        student_logits (torch.Tensor): The logits from the student model.
        teacher_logits (torch.Tensor): The logits from the teacher model.
        labels (torch.Tensor): The true labels.
        alpha (float): The weight for the distillation loss.
        T (float): The temperature for sigmoid.
        
        
    Returns:
        torch.Tensor: The combined distillation loss.
    """
    # soft label
    teacher_prob = torch.sigmoid(teacher_logits / T)
    student_prob = torch.sigmoid(student_logits / T)
    distill_loss = F.mse_loss(student_prob, teacher_prob)

    # hard label
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

    # combine
    return alpha * distill_loss + (1 - alpha) * hard_loss


########################################
# 2) Distillation single epoch         #
########################################
def distill_train_one_epoch(teacher_model, student_model, loader, optimizer, device,
                            alpha=0.5, temperature=2.0):
    teacher_model.eval()
    student_model.train()

    total_loss = 0.0
    for data in loader:
        data = data.to(device)

        # teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(data)  # => shape=(1,1)

        # student forward
        student_out = student_model(data)      # => shape=(1,1)

        teacher_logits = teacher_out.view(-1)
        student_logits = student_out.view(-1)
        label = data.y.float().to(device).view(-1)

        # distill loss
        loss = distill_loss_fn(student_logits, teacher_logits, label,
                               alpha=alpha, T=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


########################################
# 3) Student pure BCE training         #
########################################
# This could probably be merged just training.
def student_train_one_epoch(student_model, loader, optimizer, device):
    """
    Calculates the distillation loss using both soft and hard labels.
    
    Args:
        student_logits (torch.Tensor): The logits from the student model.
        teacher_logits (torch.Tensor): The logits from the teacher model.
        labels (torch.Tensor): The true labels.
        alpha (float): The weight for the distillation loss.
        T (float): The temperature for sigmoid.
        
        
    Returns:
        torch.Tensor: The combined distillation loss.
    """
    student_model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)

        out = student_model(data).view(-1)
        label = data.y.float().to(device).view(-1)

        loss = F.binary_cross_entropy_with_logits(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)