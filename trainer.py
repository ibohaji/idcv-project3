import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Dict, List, Callable, Any
import numpy as np

from infra.metrics import dice_score, iou_score, accuracy, sensitivity, specificity
from infra.losses import FocalLoss


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
        for batch_idx, batch in enumerate(pbar):
            # Support datasets that optionally return an additional ROI/FOV mask
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, masks, roi_mask = batch
            else:
                images, masks = batch
                roi_mask = masks  # fallback: use lesion mask as weighting mask

            images = images.to(self.device)
            # Handle mask format - ensure it's float and correct shape
            if masks.dim() == 3:  # Add channel dimension if missing
                masks = masks.unsqueeze(1)
            masks = masks.to(self.device).float()

            if roi_mask.dim() == 3:
                roi_mask = roi_mask.unsqueeze(1)
            roi_mask = roi_mask.to(self.device).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Support losses that take an additional mask argument (e.g., FocalLoss with ROI/pos mask)
            if isinstance(self.loss_fn, FocalLoss):
                loss = self.loss_fn(outputs, masks, roi_mask)
            else:
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

        # For FocalLoss: accumulate full loss vs probability statistics per bin (no subsampling)
        use_focal_bins = isinstance(self.loss_fn, FocalLoss)
        if use_focal_bins:
            num_bins = 50
            bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=self.device)
            bin_sums = torch.zeros(num_bins, device=self.device)
            bin_counts = torch.zeros(num_bins, device=self.device)
        
        with torch.no_grad():
            desc = f"{desc_prefix}Evaluating" if desc_prefix else "Evaluating"
            pbar = tqdm(dataloader, desc=desc)
            for batch in pbar:
                # Support datasets that optionally return an additional ROI/FOV mask
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    images, masks, roi_mask = batch
                else:
                    images, masks = batch
                    roi_mask = masks  # fallback: use lesion mask as weighting mask

                images = images.to(self.device)
                # Handle mask format - ensure it's float and correct shape
                if masks.dim() == 3:  # Add channel dimension if missing
                    masks = masks.unsqueeze(1)
                masks = masks.to(self.device).float()

                if roi_mask.dim() == 3:
                    roi_mask = roi_mask.unsqueeze(1)
                roi_mask = roi_mask.to(self.device).float()
                
                outputs = self.model(images)

                # Support losses that take an additional mask argument (e.g., FocalLoss with ROI/pos mask)
                if use_focal_bins:
                    loss = self.loss_fn(outputs, masks, roi_mask)

                    # Additionally, accumulate full (p_t, focal_loss) statistics per bin for analysis/plotting
                    p = torch.sigmoid(outputs)
                    # probability of the true class
                    p_t = p * masks + (1 - p) * (1 - masks)

                    # base CE term
                    ce = -torch.log(p_t.clamp_min(1e-8))

                    alpha = self.loss_fn.alpha
                    gamma = self.loss_fn.gamma
                    alpha_t = alpha * masks + (1 - alpha) * (1 - masks)
                    focal_per_pixel = alpha_t * (1 - p_t) ** gamma * ce

                    # flatten and bin without subsampling
                    p_t_flat = p_t.view(-1)
                    focal_flat = focal_per_pixel.view(-1)

                    if p_t_flat.numel() > 0:
                        # bin indices: 0..num_bins-1
                        indices = torch.bucketize(p_t_flat, bin_edges) - 1
                        indices = indices.clamp(min=0, max=num_bins - 1)
                        # accumulate sums and counts
                        bin_sums.scatter_add_(0, indices, focal_flat)
                        bin_counts.scatter_add_(0, indices, torch.ones_like(focal_flat))
                else:
                    loss = self.loss_fn(outputs, masks)
                
                total_loss += loss.item()
                
                for name, metric_fn in self.metrics.items():
                    metric_val = metric_fn(outputs, masks).item()
                    metric_values[name].append(metric_val)
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {name: np.mean(vals) for name, vals in metric_values.items()}

        result: Dict[str, Any] = {'loss': avg_loss, **avg_metrics}

        # Attach binned probability / loss statistics for FocalLoss if available
        if use_focal_bins and bin_counts.sum() > 0:
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            mean_loss = bin_sums / (bin_counts + 1e-8)
            result["focal_bin_centers"] = bin_centers.cpu().tolist()
            result["focal_loss_by_bin"] = mean_loss.cpu().tolist()
        
        return result

