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
    def __init__(self, student, teacher=None, device="cuda", **kwargs):
        self.student = student.to(device)
        self.teacher = teacher.to(device) if teacher else None
        self.device = device
        
        # Distillation-specific parameters
        self.distill_alpha = kwargs.get('distill_alpha', 0.5)
        self.warmup_epochs = kwargs.get('warmup_epochs', 5)
        self.teacher_epochs = kwargs.get('teacher_epochs', 10)
        self.student_epochs = kwargs.get('student_epochs', 20)

    def train_teacher(self, train_loader):
        """Train the teacher model using PyTorchTrainer."""
        if not self.teacher:
            raise ValueError("Teacher model is not defined.")
        
        teacher_trainer = PyTorchTrainer(
            model=self.teacher,
            optimizer=torch.optim.Adam(self.teacher.parameters()),
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            device=self.device
        )
        
        for epoch in range(self.teacher_epochs):
            teacher_trainer.train_one_epoch(train_loader)
            teacher_trainer.validate(train_loader)  # Add validation step here
            metrics = teacher_trainer.report_latest_metrics()
            print(f"Teacher Epoch {epoch+1} | Loss: {metrics['train']['loss']:.4f}")

    def train_student(self, train_loader):
        """Train the student model using PyTorchDistillationTrainer."""
        student_trainer = PyTorchDistillationTrainer(
            model=self.student,
            optimizer=torch.optim.Adam(self.student.parameters()),
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            teacher_model=self.teacher,
            warmup_epochs=self.warmup_epochs,
            alpha=self.distill_alpha,
            device=self.device
        )
        
        for epoch in range(self.student_epochs):
            student_trainer.train_one_epoch(train_loader)
            student_trainer.validate(train_loader)
            metrics = student_trainer.report_latest_metrics()
            print(f"Student Epoch {epoch+1} | Loss: {metrics['train']['loss']:.4f} | Acc: {metrics['train']['accuracy']:.2f}")

    def train_sequential(self, train_loader):
        """Train the teacher first, then the student."""
        if self.teacher:
            print("Training teacher model...")
            self.train_teacher(train_loader)
        
        print("Training student model...")
        self.train_student(train_loader)

class TrainingOrchestrator:
    def __init__(self, teacher_model=None, student_model=None, device="cuda", **kwargs):
        self.trainer = DistillationTrainer(
            student=student_model,
            teacher=teacher_model,
            device=device,
            **kwargs
        )

    def train_teacher(self, train_loader):
        self.trainer.train_teacher(train_loader)

    def train_student(self, train_loader):
        self.trainer.train_student(train_loader)

    def train_sequential(self, train_loader):
        self.trainer.train_sequential(train_loader)

