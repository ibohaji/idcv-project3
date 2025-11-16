"""Script to visualize model predictions on test samples."""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.settings import load_settings_from_yaml, load_parameters_from_yaml
from data.dataloaders import PH2Dataset, DRIVEDataset
from models import EncoderDecoder, Unet
from infra.losses import BCE, FocalLoss, WeightedBCE


def load_model(checkpoint_path: Path, model_name: str, in_channels: int = 3):
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model configuration from checkpoint
    if 'best_config' in checkpoint:
        config = checkpoint['best_config']
        model_name = config.get('model', model_name)
        in_channels = config.get('in_channels', in_channels)
    
    # Create model
    if model_name.lower() in ['encoderdecoder', 'decoder_encoder']:
        model = EncoderDecoder(in_channels=in_channels)
    elif model_name.lower() == 'unet':
        model = Unet(in_channels=in_channels, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('best_config', {})


def get_dataloader(dataset_name: str, dataset_path: str, model_name: str, settings, split='val'):
    """Get dataloader for a dataset (train/val/test)."""
    # Get normalization stats
    norm_stats = settings.normalization.get(dataset_name, {
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
    
    # Test transforms (no augmentation)
    transform = A.Compose([
        A.Resize(resize_h, resize_w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    # Create dataset
    if dataset_name == 'PH2':
        dataset = PH2Dataset(split=split, transform=transform, path=dataset_path)
    elif dataset_name == 'DRIVE':
        dataset = DRIVEDataset(split=split, transform=transform, path=dataset_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return loader


def denormalize_image(image_tensor, mean, std):
    """Denormalize image tensor for visualization."""
    image = image_tensor.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    image = torch.clamp(image, 0, 1)
    return image


def visualize_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int = 5,
    dataset_name: str = "Unknown",
    model_name: str = "Unknown",
    loss_name: str = "Unknown",
    save_path: Path = None
):
    """Visualize model predictions on test samples."""
    model = model.to(device)
    
    # Get normalization stats for denormalization
    from config.settings import load_settings_from_yaml
    settings = load_settings_from_yaml(Path("config/settings.yaml"))
    norm_stats = settings.normalization.get(dataset_name, {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    })
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            # Get images and masks (val set has ground truth for both datasets)
            images, masks = batch_data
            images = images.to(device)
            if masks is not None:
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            predictions_binary = (predictions > 0.5).float()
            
            # Denormalize image for visualization
            image_np = denormalize_image(images[0].cpu(), mean, std)
            image_np = image_np.permute(1, 2, 0).numpy()
            
            # Get predictions
            pred_np = predictions_binary[0, 0].cpu().numpy()
            pred_prob = predictions[0, 0].cpu().numpy()
            
            # Get ground truth if available
            if masks is not None:
                gt_np = masks[0, 0].cpu().numpy()
            else:
                gt_np = None
            
            # Plot
            ax1 = axes[sample_count, 0]
            ax2 = axes[sample_count, 1]
            ax3 = axes[sample_count, 2]
            
            # Original image
            ax1.imshow(image_np)
            ax1.set_title(f'Sample {sample_count + 1}: Original Image', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Ground truth
            ax2.imshow(gt_np, cmap='gray')
            ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            # Prediction - show binary predictions to see model performance
            ax3.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            ax3.set_title(f'Binary Prediction (Threshold: 0.5)', fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            sample_count += 1
    
    plt.suptitle(
        f'{dataset_name} - {model_name} - {loss_name}\nValidation Set Predictions',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions on test samples')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of test samples to visualize (default: 5)')
    parser.add_argument('--save-dir', type=str, default='./analysis/visualizations',
                       help='Directory to save visualizations (default: ./analysis/visualizations)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file {checkpoint_path} does not exist")
        return
    
    # Load settings
    settings = load_settings_from_yaml(Path("config/settings.yaml"))
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get dataset and model info from checkpoint
    dataset_name = checkpoint.get('dataset', 'Unknown')
    best_config = checkpoint.get('best_config', {})
    model_name = best_config.get('model', 'Unknown')
    loss_name = best_config.get('loss', 'Unknown')
    optimizer_name = best_config.get('optimizer', 'Unknown')
    in_channels = best_config.get('in_channels', 3)
    
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Loss: {loss_name}")
    print(f"Optimizer: {optimizer_name}")
    
    # Load model
    model, _ = load_model(checkpoint_path, model_name, in_channels)
    
    # Get dataset path
    dataset_path = settings.dataset_paths.get(dataset_name)
    if dataset_path is None:
        print(f"Error: No path configured for dataset: {dataset_name}")
        return
    
    # Get validation dataloader (has ground truth)
    print(f"Loading validation dataset from: {dataset_path}")
    val_loader = get_dataloader(dataset_name, dataset_path, model_name, settings, split='val')
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    filename = f"{dataset_name}_{model_name}_{loss_name}_{optimizer_name}_predictions.png"
    save_path = save_dir / filename
    
    # Visualize
    print(f"\nGenerating visualizations for {args.num_samples} samples...")
    visualize_predictions(
        model=model,
        dataloader=val_loader,
        device=args.device,
        num_samples=min(args.num_samples, len(val_loader.dataset)),
        dataset_name=dataset_name,
        model_name=model_name,
        loss_name=loss_name,
        save_path=save_path
    )
    
    print(f"\nVisualization saved to: {save_path}")


if __name__ == '__main__':
    main()

