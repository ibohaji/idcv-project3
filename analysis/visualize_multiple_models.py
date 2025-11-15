"""Script to visualize predictions from multiple models side-by-side for comparison."""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.settings import load_settings_from_yaml
from data.dataloaders import PH2Dataset, DRIVEDataset
from models import EncoderDecoder, Unet
from analysis.visualize_predictions import load_model, get_test_dataloader, denormalize_image


def visualize_multiple_models(
    checkpoints: list,
    dataloader: DataLoader,
    device: str,
    num_samples: int = 3,
    dataset_name: str = "Unknown",
    save_path: Path = None
):
    """Visualize predictions from multiple models side-by-side."""
    models = []
    model_configs = []
    
    # Load all models
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        best_config = checkpoint.get('best_config', {})
        model_name = best_config.get('model', 'Unknown')
        in_channels = best_config.get('in_channels', 3)
        
        model, _ = load_model(checkpoint_path, model_name, in_channels)
        model = model.to(device)
        model.eval()
        
        models.append(model)
        model_configs.append({
            'model': model_name,
            'loss': best_config.get('loss', 'Unknown'),
            'optimizer': best_config.get('optimizer', 'Unknown'),
            'dice': checkpoint.get('best_val_dice', 0.0)
        })
    
    # Get normalization stats
    settings = load_settings_from_yaml(Path("config/settings.yaml"))
    norm_stats = settings.normalization.get(dataset_name, {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    })
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    
    # Create figure: rows = samples, cols = 1 (image) + 1 (GT) + N (models)
    n_models = len(models)
    fig, axes = plt.subplots(num_samples, 2 + n_models, figsize=(5 * (2 + n_models), 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            # Handle test set
            if dataset_name == 'DRIVE' and isinstance(batch_data, torch.Tensor):
                images = batch_data.to(device)
                masks = None
            else:
                images, masks = batch_data
                images = images.to(device)
                if masks is not None:
                    if masks.dim() == 3:
                        masks = masks.unsqueeze(1)
                    masks = masks.to(device)
            
            # Denormalize image
            image_np = denormalize_image(images[0].cpu(), mean, std)
            image_np = image_np.permute(1, 2, 0).numpy()
            
            # Get ground truth
            if masks is not None:
                gt_np = masks[0, 0].cpu().numpy()
            else:
                gt_np = None
            
            # Plot original image
            ax = axes[sample_count, 0]
            ax.imshow(image_np)
            ax.set_title('Original Image', fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # Plot ground truth
            ax = axes[sample_count, 1]
            if gt_np is not None:
                ax.imshow(gt_np, cmap='gray')
                ax.set_title('Ground Truth', fontsize=11, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No GT\n(Test Set)', ha='center', va='center', fontsize=12)
                ax.set_title('Ground Truth (N/A)', fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # Get predictions from each model
            for model_idx, model in enumerate(models):
                outputs = model(images)
                predictions = torch.sigmoid(outputs)
                predictions_binary = (predictions > 0.5).float()
                pred_np = predictions_binary[0, 0].cpu().numpy()
                pred_prob = predictions[0, 0].cpu().numpy()
                
                config = model_configs[model_idx]
                title = f"{config['model']}\n{config['loss']} + {config['optimizer']}\nDice: {config['dice']:.3f}"
                
                ax = axes[sample_count, 2 + model_idx]
                ax.imshow(pred_np, cmap='gray')
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.axis('off')
            
            sample_count += 1
    
    plt.suptitle(
        f'{dataset_name} - Model Comparison\nTest Set Predictions',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize predictions from multiple models')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints (.pth files)')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of test samples to visualize (default: 3)')
    parser.add_argument('--save-dir', type=str, default='./analysis/visualizations',
                       help='Directory to save visualizations (default: ./analysis/visualizations)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    # Validate checkpoints
    checkpoint_paths = [Path(cp) for cp in args.checkpoints]
    for cp in checkpoint_paths:
        if not cp.exists():
            print(f"Error: Checkpoint file {cp} does not exist")
            return
    
    # Load settings
    settings = load_settings_from_yaml(Path("config/settings.yaml"))
    
    # Group checkpoints by dataset
    checkpoints_by_dataset = {}
    for cp_path in checkpoint_paths:
        checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
        dataset_name = checkpoint.get('dataset', 'Unknown')
        if dataset_name not in checkpoints_by_dataset:
            checkpoints_by_dataset[dataset_name] = []
        checkpoints_by_dataset[dataset_name].append(cp_path)
    
    print(f"Found checkpoints for {len(checkpoints_by_dataset)} dataset(s): {list(checkpoints_by_dataset.keys())}")
    print(f"This will generate {len(checkpoints_by_dataset)} separate comparison plot(s), one per dataset.\n")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize each dataset separately
    for dataset_name, dataset_checkpoints in checkpoints_by_dataset.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name} ({len(dataset_checkpoints)} models)")
        print(f"{'='*60}")
        
        # Get dataset path
        dataset_path = settings.dataset_paths.get(dataset_name)
        if dataset_path is None:
            print(f"Error: No path configured for dataset: {dataset_name}")
            continue
        
        # Get model name from first checkpoint for dataloader (all should be same dataset)
        checkpoint = torch.load(dataset_checkpoints[0], map_location='cpu', weights_only=False)
        best_config = checkpoint.get('best_config', {})
        model_name = best_config.get('model', 'EncoderDecoder')
        
        # Get test dataloader
        print(f"Loading test dataset from: {dataset_path}")
        test_loader = get_test_dataloader(dataset_name, dataset_path, model_name, settings)
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Generate filename
        filename = f"{dataset_name}_model_comparison.png"
        save_path = save_dir / filename
        
        # Visualize
        print(f"\nGenerating comparison visualizations for {args.num_samples} samples...")
        visualize_multiple_models(
            checkpoints=dataset_checkpoints,
            dataloader=test_loader,
            device=args.device,
            num_samples=min(args.num_samples, len(test_loader.dataset)),
            dataset_name=dataset_name,
            save_path=save_path
        )
        
        print(f"\nComparison saved to: {save_path}")


if __name__ == '__main__':
    main()

