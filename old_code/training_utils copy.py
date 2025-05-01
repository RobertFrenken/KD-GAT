import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocessing import graph_creation, GraphDataset
from models.models import GATWithJK
import hydra
#########################################
# 1) PyTorch Trainer Class              #
#########################################
class PyTorchTrainer:
    def __init__(self, model, optimizer, loss_fn, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Training state tracking
        self.best_val_loss = float('inf')
        self.best_model_weights = None
        self.metrics = {
            'train': {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []},
            'val': {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        }
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        all_preds, all_targets = [], []
        
        for batch in train_loader:
            batch = batch.to(self.device)
            targets = batch.y.float()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch).squeeze()
            loss = self.loss_fn(outputs, batch.y.float())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics collection
            epoch_loss += loss.item() * batch.size(0)
            preds = (outputs > 0).long()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Update training metrics
        self._update_metrics('train', epoch_loss/len(train_loader.dataset), 
                            all_preds, all_targets)

    def validate(self, val_loader):
        self.model.eval()
        epoch_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                targets = batch.y.float()
                
                outputs = self.model(batch).squeeze()
                loss = self.loss_fn(outputs, targets)
                
                epoch_loss += loss.item() * batch.size(0)
                preds = (outputs > 0).long()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Update validation metrics
        avg_loss = epoch_loss / len(val_loader.dataset)
        self._update_metrics('val', avg_loss, all_preds, all_targets)
        
        # Check for best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_model_weights = self.model.state_dict().copy()

    def compute_metrics(self, phase):
        return self.metrics[phase]

    def _update_metrics(self, phase, loss, preds, targets):
        self.metrics[phase]['loss'].append(loss)
        self.metrics[phase]['accuracy'].append(accuracy_score(targets, preds))
        self.metrics[phase]['f1'].append(f1_score(targets, preds, average='macro'))
        self.metrics[phase]['precision'].append(
            precision_score(targets, preds, average='macro', zero_division=0)
        )
        self.metrics[phase]['recall'].append(
            recall_score(targets, preds, average='macro', zero_division=0)
        )

    def report_latest_metrics(self):
        return {
            'train': {k: v[-1] for k, v in self.metrics['train'].items()},
            'val': {k: v[-1] for k, v in self.metrics['val'].items()}
        }

class PyTorchDistillationTrainer(PyTorchTrainer):
    def __init__(self, model, optimizer, loss_fn, device="cuda",
                 teacher_model=None, warmup_epochs=5, alpha=0.5,
                 distillation_loss_fn=torch.nn.KLDivLoss(reduction='batchmean')):
        super().__init__(model, optimizer, loss_fn, device)
        
        # Distillation-specific parameters
        self.teacher_model = teacher_model
        if self.teacher_model:
            self.teacher_model = teacher_model.to(device).eval()
            
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.distillation_loss_fn = distillation_loss_fn
        self.current_epoch = 0
        
        # Add distillation metrics
        self.metrics['train']['student_loss'] = []
        self.metrics['train']['distill_loss'] = []

    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_student_loss = 0
        epoch_distill_loss = 0
        all_preds, all_targets = [], []
        
        for batch in train_loader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            student_outputs = self.model(inputs)
            
            if self.teacher_model and self.current_epoch >= self.warmup_epochs:
                # Distillation phase
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs)
                
                # Calculate combined loss
                student_loss = self.loss_fn(student_outputs, targets)
                distill_loss = self.distillation_loss_fn(
                    torch.log_softmax(student_outputs, dim=-1),
                    torch.softmax(teacher_outputs, dim=-1)
                )
                loss = (1 - self.alpha) * student_loss + self.alpha * distill_loss
                
                # Track both losses
                epoch_student_loss += student_loss.item() * inputs.size(0)
                epoch_distill_loss += distill_loss.item() * inputs.size(0)
            else:
                # Warmup phase or normal training
                loss = self.loss_fn(student_outputs, targets)
                epoch_student_loss += loss.item() * inputs.size(0)
                epoch_distill_loss = 0  # Not applicable
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track predictions
            preds = (student_outputs > 0).long()
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        
        # Update metrics
        total_samples = len(train_loader.dataset)
        self.metrics['train']['student_loss'].append(epoch_student_loss / total_samples)
        if self.teacher_model and self.current_epoch >= self.warmup_epochs:
            self.metrics['train']['distill_loss'].append(epoch_distill_loss / total_samples)
        
        self._update_metrics('train', 
                            (epoch_student_loss + epoch_distill_loss) / total_samples,
                            all_preds, all_targets)
        
        self.current_epoch += 1

    def validate(self, val_loader):
        # Validation remains the same - only student model is evaluated
        return super().validate(val_loader)

    def report_latest_metrics(self):
        metrics = super().report_latest_metrics()
        if self.teacher_model and self.current_epoch > self.warmup_epochs:
            metrics['train']['student_loss'] = self.metrics['train']['student_loss'][-1]
            metrics['train']['distill_loss'] = self.metrics['train']['distill_loss'][-1]
        return metrics

class DistillationTrainer:
    def __init__(self, student, teacher, device="cuda", **kwargs):
        self.student = hydra.utils.instantiate(student).to(device)
        self.teacher = hydra.utils.instantiate(teacher).to(device) if teacher else None
        self.device = device
        
        # Initialize from config
        self.mode = kwargs.get('mode', 'sequential')
        self.distill_alpha = kwargs.get('distill_alpha', 0.5)
        self.warmup_epochs = kwargs.get('warmup_epochs', 5)

    def train(self, train_loader, teacher_epochs=10, student_epochs=20):
        if self.mode in ['teacher', 'sequential']:
            self._train_teacher(train_loader, teacher_epochs)
            
        if self.mode in ['student', 'sequential']:
            self._train_student(train_loader, student_epochs)

    def _train_teacher(self, train_loader, epochs):
        optimizer = torch.optim.Adam(self.teacher.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Standard training loop
            self.teacher.train()
            for batch in train_loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.teacher(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def _train_student(self, train_loader, epochs):
        optimizer = torch.optim.Adam(self.student.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        distill_loss = torch.nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(epochs):
            self.student.train()
            if self.teacher: 
                self.teacher.eval()
            
            for batch in train_loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                
                student_out = self.student(inputs)
                
                if self.teacher and epoch >= self.warmup_epochs:
                    with torch.no_grad():
                        teacher_out = self.teacher(inputs)
                    
                    loss = (1-self.distill_alpha)*criterion(student_out, targets) + \
                           self.distill_alpha*distill_loss(
                               torch.log_softmax(student_out, dim=1),
                               torch.softmax(teacher_out, dim=1)
                           )
                else:
                    loss = criterion(student_out, targets)
                
                loss.backward()
                optimizer.step()

class TrainingOrchestrator:
    def __init__(self, teacher_model=None, student_model=None):
        self.teacher = teacher_model
        self.student = student_model
        
    def train_teacher(self, train_loader, epochs, **kwargs):
        print("Training teacher model...")
        teacher_trainer = PyTorchTrainer(
            model=self.teacher,
            optimizer=torch.optim.Adam(self.teacher.parameters()),
            loss_fn=torch.nn.CrossEntropyLoss()
        )
        
        for epoch in range(epochs):
            teacher_trainer.train_one_epoch(train_loader)
            print(f"Teacher Epoch {epoch+1} Loss: {teacher_trainer.report_latest_metrics()['train']['loss']:.4f}")

    def train_student(self, train_loader, epochs, teacher=None, **kwargs):
        print("Training student model...")
        student_trainer = PyTorchDistillationTrainer(
            model=self.student,
            optimizer=torch.optim.Adam(self.student.parameters()),
            loss_fn=torch.nn.CrossEntropyLoss(),
            teacher_model=teacher
        )
        
        for epoch in range(epochs):
            student_trainer.train_one_epoch(train_loader)
            metrics = student_trainer.report_latest_metrics()
            print(f"Student Epoch {epoch+1} | Loss: {metrics['train']['loss']:.4f} | Acc: {metrics['train']['accuracy']:.2f}")

    def train_sequential(self, train_loader, teacher_epochs, student_epochs, **kwargs):
        print("Sequential training: Teacher -> Student")
        self.train_teacher(train_loader, teacher_epochs)
        self.train_student(train_loader, student_epochs, teacher=self.teacher)
#####################################################################

def evaluation(loader, model, device, desc="[Model]", 
              print_preds=False):
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
    if print_preds:
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

        epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_epoch_loss = evaluation(test_loader, model, device, desc="[Model]", print_preds=False)
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
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Completes the training step of one epoch
    
    Args:
    model (torch.nn.Module): The model to be trained.
    loader (torch_geometric.loader.DataLoader): DataLoader for the dataset.
    optimizer (torch.optim.Optimizer): The optimizer for the model.
    criterion (torch.nn.Module): The loss function.
    device (torch.device): The device to run the model on.
        
    Returns:
        epoch_loss: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        batch.to(device) # put batch tensor on the correct device
        out = model(batch).squeeze()
        loss = criterion(out, batch.y.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch.size(0)  # Multiply by batch size to get total loss for the batch

    epoch_loss = running_loss / len(loader.dataset)  # Average loss over the entire dataset
    return  epoch_loss