"""Create a clean segmentation visualization figure for DRIVE dataset suitable for report."""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.settings import load_settings_from_yaml
from data.dataloaders import DRIVEDataset
from models import Unet
from analysis.visualize_predictions import denormalize_image


def load_model(checkpoint_path: Path, model_name: str = 'Unet', in_channels: int = 3):
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model configuration from checkpoint
    if 'best_config' in checkpoint:
        config = checkpoint['best_config']
        model_name = config.get('model', model_name)
        in_channels = config.get('in_channels', in_channels)
    
    # Create model
    if model_name.lower() == 'unet':
        model = Unet(in_channels=in_channels, out_channels=1)
    else:
        raise ValueError(f"Model {model_name} not supported for this visualization")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('best_config', {})


def create_segmentation_figure(
    checkpoint_path: Path,
    dataset_path: str,
    num_samples: int = 3,
    save_path: Path = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Create a clean segmentation visualization figure for the report."""
    
    # Load settings
    settings = load_settings_from_yaml(Path("config/settings.yaml"))
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model info
    best_config = checkpoint.get('best_config', {})
    model_name = best_config.get('model', 'Unet')
    in_channels = best_config.get('in_channels', 3)
    dice_score = checkpoint.get('best_val_dice', 0.0)
    
    print(f"Model: {model_name}, Dice: {dice_score:.4f}")
    
    # Load model
    model, _ = load_model(checkpoint_path, model_name, in_channels)
    model = model.to(device)
    
    # Get normalization stats
    norm_stats = settings.normalization.get('DRIVE', {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    })
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    
    # Create validation dataloader (no augmentation)
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    dataset = DRIVEDataset(split='val', transform=transform, path=dataset_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Create figure: 3 columns (Image, GT, Prediction) x num_samples rows
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if sample_count >= num_samples:
                break
            
            # Get image, mask, and roi_mask
            image, mask, roi_mask = batch_data
            image = image.to(device)
            mask = mask.to(device)
            roi_mask = roi_mask.to(device)
            
            # Get prediction
            output = model(image)
            prediction = torch.sigmoid(output)
            prediction_binary = (prediction > 0.5).float()
            
            # Denormalize image
            image_np = denormalize_image(image[0].cpu(), mean, std)
            image_np = image_np.permute(1, 2, 0).numpy()
            image_np = np.clip(image_np, 0, 1)
            
            # Get masks
            gt_np = mask[0, 0].cpu().numpy()
            pred_np = prediction_binary[0, 0].cpu().numpy()
            roi_np = roi_mask[0, 0].cpu().numpy() if roi_mask is not None else None
            
            # Apply ROI mask to predictions for visualization
            if roi_np is not None:
                pred_np = pred_np * roi_np
                gt_np = gt_np * roi_np
            
            # Plot
            ax1, ax2, ax3 = axes[sample_count, 0], axes[sample_count, 1], axes[sample_count, 2]
            
            # Original image
            ax1.imshow(image_np)
            ax1.set_title('Original Image', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Ground truth
            ax2.imshow(gt_np, cmap='gray', vmin=0, vmax=1)
            ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            # Prediction
            ax3.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            ax3.set_title('Prediction', fontsize=14, fontweight='bold')
            ax3.axis('off')
            
            sample_count += 1
    
    # Add overall title
    loss_name = best_config.get('loss', 'Unknown')
    optimizer_name = best_config.get('optimizer', 'Unknown')
    fig.suptitle(
        f'DRIVE Segmentation Results: {model_name} + {loss_name} + {optimizer_name} (Dice: {dice_score:.4f})',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create DRIVE segmentation visualization figure for report'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to DRIVE dataset (default: from settings.yaml)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of samples to visualize (default: 3)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='images/DRIVE_segmentation_results.png',
        help='Path to save the figure (default: images/DRIVE_segmentation_results.png)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available, else cpu)'
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file {checkpoint_path} does not exist")
        return
    
    # Get dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        settings = load_settings_from_yaml(Path("config/settings.yaml"))
        dataset_path = settings.dataset_paths.get('DRIVE')
        if dataset_path is None:
            print("Error: No DRIVE dataset path found. Specify --dataset-path")
            return
    
    save_path = Path(args.save_path)
    
    create_segmentation_figure(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        num_samples=args.num_samples,
        save_path=save_path,
        device=args.device
    )


if __name__ == '__main__':
    main()

