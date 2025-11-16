import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.settings import Settings, Parameters
from data.dataloaders import PH2Dataset, DRIVEDataset
from models import EncoderDecoder, Unet
from infra.losses import BCE, FocalLoss, WeightedBCE


class ExperimentTracker:
    """Handles saving experiment results in a structured, dataset-centric format."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.output_dir = Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_dataset_results(
        self,
        experiment_name: str,
        dataset_name: str,
        experiments: List[Dict[str, Any]]
    ) -> Path:
        """
        Save all experiment results for a dataset to a single JSON file.
        
        Args:
            experiment_name: Name of the experiment study
            dataset_name: Name of the dataset
            experiments: List of experiment results, each containing:
                - model: Model name
                - loss: Loss function name
                - optimizer: Optimizer name
                - results: Full experiment results dict
        
        Returns:
            Path to saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{dataset_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        dataset_data = {
            'experiment_name': experiment_name,
            'dataset': dataset_name,
            'timestamp': timestamp,
            'total_experiments': len(experiments),
            'experiments': experiments
        }
        
        with open(filepath, 'w') as f:
            json.dump(dataset_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Saved {len(experiments)} experiments for {dataset_name} to:")
        print(f"{filepath}")
        print(f"{'='*60}\n")
        return filepath


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        settings: Settings,
        parameters: Parameters,
        tracker: ExperimentTracker
    ):
        self.experiment_name = experiment_name
        self.settings = settings
        self.parameters = parameters
        self.tracker = tracker
    
    @staticmethod
    def get_model(model_name: str, in_channels: int = 3):
        """Get model by name"""
        if model_name.lower() in ['encoderdecoder', 'decoder_encoder']:
            return EncoderDecoder(in_channels=in_channels)
        elif model_name.lower() == 'unet':
            return Unet(in_channels=in_channels, out_channels=1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def get_loss(loss_name: str):
        """Get loss function by name (supports Focal with different gammas via name suffix).
        
        Examples:
            'Focal'      -> FocalLoss(gamma=2)  # default
            'Focal1'     -> FocalLoss(gamma=1)
            'Focal_3'    -> FocalLoss(gamma=3)
            'focal-g5'   -> FocalLoss(gamma=5)
        """
        raw_name = loss_name
        loss_name = loss_name.lower()
        if loss_name in ['bce', 'cross_entropy']:
            return BCE()
        elif loss_name.startswith('focal'):
            # Default gamma
            gamma = 2
            # Extract any digits in the name as gamma if present
            digits = ''.join(ch for ch in loss_name if ch.isdigit())
            if digits:
                gamma = int(digits)
            print(f"Using FocalLoss with gamma={gamma} for loss='{raw_name}'")
            return FocalLoss(gamma=gamma)
        elif loss_name == 'weightedbce':
            return WeightedBCE()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def get_transforms(self, dataset_name: str, model_name: str = None, is_training: bool = True):
        """Get transforms - resize, augment (if training), and normalize using dataset-specific mean/std from settings"""
        # Get normalization values from settings
        norm_stats = self.settings.normalization.get(dataset_name, {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        })
        mean = norm_stats["mean"]
        std = norm_stats["std"]
        
        # Determine resize dimensions
        if model_name and model_name.lower() == 'encoderdecoder':
            resize_h, resize_w = 200, 200
        else:
            resize_h, resize_w = 256, 256
        
        # Base transforms: resize and normalize
        transforms = [
            A.Resize(resize_h, resize_w),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
        
        # Add augmentation only for training
        if is_training:
            augmentation = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
            # Insert augmentation before normalization
            transforms = [A.Resize(resize_h, resize_w)] + augmentation + [A.Normalize(mean=mean, std=std), ToTensorV2()]
        
        # Support an optional second mask (e.g., FOV / ROI) with the same transforms
        return A.Compose(transforms, additional_targets={'mask2': 'mask'})
    
    def get_dataloaders(self, dataset_name: str, model_name: str = None):
        """Get train and val dataloaders for a dataset"""
        train_transform = self.get_transforms(dataset_name, model_name, is_training=True)
        val_transform = self.get_transforms(dataset_name, model_name, is_training=False)
        
        # Get dataset path from settings
        dataset_path = self.settings.dataset_paths.get(dataset_name)
        if dataset_path is None:
            raise ValueError(f"No path configured for dataset: {dataset_name}")
        
        if dataset_name == 'PH2':
            train_dataset = PH2Dataset(split='train', transform=train_transform, path=dataset_path)
            val_dataset = PH2Dataset(split='val', transform=val_transform, path=dataset_path)
            in_channels = 3  # PH2 images are RGB
        else:  # DRIVE
            # DRIVE: use full training set for training, official test set for validation
            train_dataset = DRIVEDataset(split='train', transform=train_transform, path=dataset_path)
            val_dataset = DRIVEDataset(split='test', transform=val_transform, path=dataset_path)
            in_channels = 3
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.parameters.batch_size,
            shuffle=True,
            num_workers=self.settings.num_workers,
            pin_memory=self.settings.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.parameters.batch_size,
            shuffle=False,
            num_workers=self.settings.num_workers,
            pin_memory=self.settings.pin_memory
        )
        
        return train_loader, val_loader, in_channels
    
    def run_all(self, models: List[str], datasets: List[str], losses: List[str], optimizers: List[str]):
        """
        Run all experiment combinations and save results per dataset.
        
        For each dataset, collects all model/loss/optimizer combinations and saves
        them to a single JSON file with a structured, nested format.
        Only saves the best checkpoint per dataset (not per experiment).
        """
        for dataset_name in datasets:
            # Check if results already exist for this dataset
            existing_results = list(self.tracker.output_dir.glob(f"{self.experiment_name}_{dataset_name}_*.json"))
            if existing_results:
                print(f"\n{'='*60}\nDataset: {dataset_name} - SKIPPING (results already exist)\n{'='*60}")
                print(f"Found existing results: {existing_results[0].name}")
                continue
            
            print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
            
            # Collect all experiments for this dataset
            dataset_experiments = []
            
            # Track best model for this dataset (for checkpoint saving)
            # Use sensitivity (recall for positive class) as the primary metric
            best_dataset_state = None
            best_dataset_sensitivity = 0.0
            best_dataset_loss = float('inf')
            best_dataset_dice = 0.0
            best_dataset_config = None
            
            for model_name in models:
                print(f"\nModel: {model_name}")
                
                # Get dataloaders with model-specific transforms
                train_loader, val_loader, in_channels = self.get_dataloaders(dataset_name, model_name)
                print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
                
                for optimizer_name in optimizers:
                    print(f"  Optimizer: {optimizer_name}")
                    
                    for loss_name in losses:
                        print(f"    Loss: {loss_name}")
                        
                        model = self.get_model(model_name, in_channels)
                        loss_fn = self.get_loss(loss_name)
                        
                        results = self.run(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            loss_fn=loss_fn,
                            optimizer_name=optimizer_name,
                            dataset_name=dataset_name,
                            model_name=model_name,
                            loss_name=loss_name
                        )
                        
                        print(
                            f"      Best Epoch: {results['best_epoch']}, "
                            f"Sensitivity: {results.get('best_val_sensitivity', 0.0):.4f}, "
                            f"Dice: {results['best_val_dice']:.4f}"
                        )
                        
                        # Check if this is the best model for this dataset
                        # Save based on sensitivity (recall for positive class)
                        # Save immediately when we find a better one (overwrites previous checkpoint)
                        current_sens = results.get('best_val_sensitivity', 0.0)
                        if current_sens > best_dataset_sensitivity:
                            best_dataset_sensitivity = current_sens
                            best_dataset_loss = results['best_val_loss']
                            best_dataset_dice = results['best_val_dice']
                            # Save the model state dict immediately (before it gets overwritten)
                            best_dataset_state = model.state_dict().copy()
                            best_dataset_config = {
                                'model': model_name,
                                'loss': loss_name,
                                'optimizer': optimizer_name,
                                'epoch': results['best_epoch'],
                                'in_channels': in_channels
                            }
                            
                            # Save checkpoint immediately (overwrites previous best for this dataset)
                            if self.settings.save_best_only:
                                checkpoint_dir = Path(self.settings.checkpoint_dir)
                                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                                checkpoint_path = checkpoint_dir / f"{self.experiment_name}_{dataset_name}_best.pth"
                                
                                torch.save({
                                    'dataset': dataset_name,
                                    'best_config': best_dataset_config,
                                    'best_val_loss': best_dataset_loss,
                                    'best_val_sensitivity': best_dataset_sensitivity,
                                    'best_val_dice': best_dataset_dice,
                                    'model_state_dict': best_dataset_state,
                                }, checkpoint_path)
                                print(f"    → New best! Saved checkpoint: {checkpoint_path.name}")
                        
                        # Store experiment result in structured format
                        dataset_experiments.append({
                            'model': model_name,
                            'loss': loss_name,
                            'optimizer': optimizer_name,
                            'config': f"{loss_name}_{optimizer_name}",
                            'results': results
                        })
            
            # Print summary of best checkpoint for this dataset
            if self.settings.save_best_only and best_dataset_state is not None:
                checkpoint_path = Path(self.settings.checkpoint_dir) / f"{self.experiment_name}_{dataset_name}_best.pth"
                print(f"\n  Best checkpoint for {dataset_name}: {checkpoint_path.name}")
                print(f"    Config: {best_dataset_config['model']} + {best_dataset_config['loss']} + {best_dataset_config['optimizer']}")
                print(
                    f"    Val Loss: {best_dataset_loss:.4f}, "
                    f"Val Sensitivity: {best_dataset_sensitivity:.4f}, "
                    f"Val Dice: {best_dataset_dice:.4f}"
                )
            
            # Save all experiments for this dataset to a single JSON file
            self.tracker.save_dataset_results(
                self.experiment_name,
                dataset_name,
                dataset_experiments
            )
        
    def run(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer_name: str,
        dataset_name: str,
        model_name: str,
        loss_name: str
    ) -> Dict[str, Any]:
        """Run a single experiment"""
        device = torch.device(self.settings.device if torch.cuda.is_available() else 'cpu')
        
        # Setup optimizer with optimizer-specific learning rate
        optimizer_lr = self.parameters.learning_rates.get(
            optimizer_name.lower(), 
            self.parameters.learning_rates.get('adam', 0.001)  # Default fallback
        )
        
        optimizer_name_lower = optimizer_name.lower()
        if optimizer_name_lower == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=optimizer_lr,
                weight_decay=self.parameters.weight_decay
            )
        elif optimizer_name_lower == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_lr,
                weight_decay=self.parameters.weight_decay
            )
        elif optimizer_name_lower == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=optimizer_lr,
                momentum=self.parameters.momentum,
                weight_decay=self.parameters.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Setup scheduler
        if self.parameters.scheduler.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.parameters.scheduler_step_size,
                gamma=self.parameters.scheduler_gamma
            )
        else:
            scheduler = None
        
        # Setup metrics
        from infra.metrics import dice_score, iou_score, accuracy, sensitivity, specificity
        metrics = {
            'dice': dice_score,
            'iou': iou_score,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        
        # Setup trainer
        from trainer import Trainer
        trainer = Trainer(model, loss_fn, optimizer, device, metrics)
        
        # Training loop
        best_val_loss = float('inf')
        best_val_dice = 0.0
        best_val_sensitivity = 0.0
        best_epoch = 0
        start_time = time.time()
        
        train_history = []
        val_history = []
        
        # Create description prefix for progress bars
        desc_prefix = f"[{dataset_name} | {model_name} | {optimizer_name} | {loss_name}] "
        
        for epoch in range(self.parameters.epochs):
            print(f"\nEpoch {epoch+1}/{self.parameters.epochs}")
            
            # Train
            train_metrics = trainer.train_epoch(train_loader, desc_prefix=desc_prefix)
            train_history.append(train_metrics)
            
            # Validate
            val_metrics = trainer.evaluate(val_loader, desc_prefix=desc_prefix)
            val_history.append(val_metrics)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Check for best model (checkpoint saving is now done per-dataset in run_all)
            # Use sensitivity (recall for positive class) as primary selection metric
            current_sens = val_metrics.get('sensitivity', 0.0)
            if current_sens > best_val_sensitivity:
                best_val_sensitivity = current_sens
                best_val_loss = val_metrics['loss']
                best_val_dice = val_metrics.get('dice', 0.0)
                best_epoch = epoch + 1
            
            # Display all metrics in a formatted table
            print(f"\n  {'Metric':<12} | {'Train':<8} | {'Val':<8}")
            print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}")
            print(f"  {'Loss':<12} | {train_metrics['loss']:>8.4f} | {val_metrics['loss']:>8.4f}")
            print(f"  {'Dice':<12} | {train_metrics.get('dice', 0):>8.4f} | {val_metrics.get('dice', 0):>8.4f}")
            print(f"  {'IoU':<12} | {train_metrics.get('iou', 0):>8.4f} | {val_metrics.get('iou', 0):>8.4f}")
            print(f"  {'Accuracy':<12} | {train_metrics.get('accuracy', 0):>8.4f} | {val_metrics.get('accuracy', 0):>8.4f}")
            print(f"  {'Sensitivity':<12} | {train_metrics.get('sensitivity', 0):>8.4f} | {val_metrics.get('sensitivity', 0):>8.4f}")
            print(f"  {'Specificity':<12} | {train_metrics.get('specificity', 0):>8.4f} | {val_metrics.get('specificity', 0):>8.4f}")
        
        total_time = time.time() - start_time
        
        # Prepare results with per-epoch history for plotting
        results = {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
            'best_val_sensitivity': best_val_sensitivity,
            'final_train_metrics': train_history[-1],
            'final_val_metrics': val_history[-1],
            'best_val_metrics': val_history[best_epoch - 1] if best_epoch > 0 else val_history[-1],
            'total_time_seconds': total_time,
            'total_epochs': self.parameters.epochs,
            'parameters': {
                'learning_rate': optimizer_lr,
                'batch_size': self.parameters.batch_size,
                'optimizer': optimizer_name,
                'scheduler': self.parameters.scheduler,
            },
            # Per-epoch history for plotting
            'train_history': train_history,
            'val_history': val_history
        }
        
        # Results are now saved per-dataset in run_all(), not individually
        return results
