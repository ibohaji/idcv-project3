"""Standalone script to calculate dataset mean and std statistics."""
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataloaders import PH2Dataset, DRIVEDataset


def calculate_mean_std(dataset, num_samples=None):
    """Calculate mean and std for a dataset."""
    means = []
    stds = []
    
    samples_to_use = num_samples if num_samples else len(dataset)
    samples_to_use = min(samples_to_use, len(dataset))
    
    for idx in tqdm(range(samples_to_use), desc="Calculating stats"):
        image, _ = dataset[idx]
        
        # Convert to numpy array
        if hasattr(image, 'mode'):  # PIL Image
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            continue
        
        # Skip if not RGB
        if len(image.shape) == 2 or image.shape[2] != 3:
            continue
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Reshape to (H*W, C)
        image = image.reshape(-1, 3)
        
        means.append(image.mean(axis=0))
        stds.append(image.std(axis=0))
    
    # Calculate overall mean and std
    mean = np.mean(means, axis=0).tolist()
    std = np.mean(stds, axis=0).tolist()
    
    return mean, std


def main():
    """Calculate and print mean/std for PH2 and DRIVE datasets"""
    
    print("="*60)
    print("PH2 Dataset Statistics")
    print("="*60)
    ph2_dataset = PH2Dataset(split='train', transform=None)
    ph2_mean, ph2_std = calculate_mean_std(ph2_dataset)
    print(f"Mean: {ph2_mean}")
    print(f"Std:  {ph2_std}")
    
    print("\n" + "="*60)
    print("DRIVE Dataset Statistics")
    print("="*60)
    drive_dataset = DRIVEDataset(split='train', transform=None)
    drive_mean, drive_std = calculate_mean_std(drive_dataset)
    print(f"Mean: {drive_mean}")
    print(f"Std:  {drive_std}")
    
    print("\n" + "="*60)
    print("Add these to config/settings.yaml:")
    print("="*60)
    print(f"normalization:")
    print(f"  PH2:")
    print(f"    mean: {ph2_mean}")
    print(f"    std: {ph2_std}")
    print(f"  DRIVE:")
    print(f"    mean: {drive_mean}")
    print(f"    std: {drive_std}")


if __name__ == "__main__":
    main()

