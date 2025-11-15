import numpy as np
from collections import Counter


def describe_basic(dataset):
    """Print basic dataset information."""
    print(f"Total samples: {len(dataset)}")
    print(f"Images: {len(dataset.images) if hasattr(dataset, 'images') else len(dataset)}")
    print(f"Masks: {len(dataset.masks) if hasattr(dataset, 'masks') else len(dataset)}")


def describe_sizes(dataset):
    """Print image and mask size statistics."""
    image_sizes = []
    mask_sizes = []
    
    for i in range(len(dataset)):
        img, mask = dataset[i]
        img_array = np.array(img)
        mask_array = np.array(mask)
        image_sizes.append((img_array.shape[0], img_array.shape[1]))
        mask_sizes.append((mask_array.shape[0], mask_array.shape[1]))
    
    image_sizes = np.array(image_sizes)
    mask_sizes = np.array(mask_sizes)
    
    print(f"\nImage sizes:")
    print(f"  Height: {image_sizes[:, 0].min()}-{image_sizes[:, 0].max()} (mean: {image_sizes[:, 0].mean():.0f})")
    print(f"  Width:  {image_sizes[:, 1].min()}-{image_sizes[:, 1].max()} (mean: {image_sizes[:, 1].mean():.0f})")
    
    print(f"\nMask sizes:")
    print(f"  Height: {mask_sizes[:, 0].min()}-{mask_sizes[:, 0].max()} (mean: {mask_sizes[:, 0].mean():.0f})")
    print(f"  Width:  {mask_sizes[:, 1].min()}-{mask_sizes[:, 1].max()} (mean: {mask_sizes[:, 1].mean():.0f})")
    
    unique_sizes = Counter([tuple(size) for size in image_sizes])
    print(f"\nUnique image sizes: {len(unique_sizes)}")
    if len(unique_sizes) <= 10:
        for size, count in unique_sizes.most_common():
            print(f"  {size}: {count} images")


def describe_structure(dataset):
    """Print data structure information."""
    img, mask = dataset[0]
    img_array = np.array(img)
    mask_array = np.array(mask)
    
    print(f"\nImage: {type(img).__name__}, shape: {img_array.shape}, dtype: {img_array.dtype}")
    print(f"  Value range: [{img_array.min()}, {img_array.max()}]")
    
    print(f"\nMask: {type(mask).__name__}, shape: {mask_array.shape}, dtype: {mask_array.dtype}")
    print(f"  Value range: [{mask_array.min()}, {mask_array.max()}]")
    print(f"  Unique values: {np.unique(mask_array)}")



