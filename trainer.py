import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Dict, List, Callable
import numpy as np

from infra.metrics import dice_score, iou_score, accuracy, sensitivity, specificity


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        metrics: Dict[str, Callable] = None
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics or {}
        
    def train_epoch(self, dataloader: DataLoader, desc_prefix: str = "") -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        metric_values = {name: [] for name in self.metrics.keys()}
        
        desc = f"{desc_prefix}Training" if desc_prefix else "Training"
        pbar = tqdm(dataloader, desc=desc)
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            # Handle mask format - ensure it's float and correct shape
            if masks.dim() == 3:  # Add channel dimension if missing
                masks = masks.unsqueeze(1)
            masks = masks.to(self.device).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            with torch.no_grad():
                for name, metric_fn in self.metrics.items():
                    metric_val = metric_fn(outputs, masks).item()
                    metric_values[name].append(metric_val)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {name: np.mean(vals) for name, vals in metric_values.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def evaluate(self, dataloader: DataLoader, desc_prefix: str = "") -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        metric_values = {name: [] for name in self.metrics.keys()}
        
        with torch.no_grad():
            desc = f"{desc_prefix}Evaluating" if desc_prefix else "Evaluating"
            pbar = tqdm(dataloader, desc=desc)
            for images, masks in pbar:
                images = images.to(self.device)
                # Handle mask format - ensure it's float and correct shape
                if masks.dim() == 3:  # Add channel dimension if missing
                    masks = masks.unsqueeze(1)
                masks = masks.to(self.device).float()
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                total_loss += loss.item()
                
                for name, metric_fn in self.metrics.items():
                    metric_val = metric_fn(outputs, masks).item()
                    metric_values[name].append(metric_val)
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {name: np.mean(vals) for name, vals in metric_values.items()}
        
        return {'loss': avg_loss, **avg_metrics}

