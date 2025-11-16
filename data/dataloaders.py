import torch
import os
import glob 
import PIL.Image as Image
import numpy as np 
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PH2Dataset(torch.utils.data.Dataset): 
    def __init__(self, split='train', transform=None, path='C:/Users/ibrahim/Documents/DTU/introduction-dplrng-cv/project3/data/datasets/PH_dataset_images/', 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        PH2 dataset for skin lesion segmentation.
        
        Args:
            split: 'train', 'val', or 'test'
            transform: Albumentations transform or None
            path: Path to PH2 dataset directory
            train_ratio: Proportion of data for training (only used if splitting internally)
            val_ratio: Proportion of data for validation (only used if splitting internally)
            test_ratio: Proportion of data for testing (only used if splitting internally)
            seed: Random seed for reproducible splits
        
        Note:
            PH2 has no predefined train/val/test splits, so we use all images for 'train'
            and 'val'/'test' will just be empty unless you manually split externally.
        """
        if transform is not None and not isinstance(transform, A.Compose):
            raise TypeError("Transform must be albumentations.Compose or None")
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        self.transform = transform 
        self.data_path = os.path.normpath(path)
        # Ensure path ends with separator for glob pattern
        if not self.data_path.endswith(os.sep):
            self.data_path += os.sep
        self.split = split
        
        # Load all images and masks
        all_masks = sorted(glob.glob(self.data_path + 'IMD*/IMD*_lesion/*.bmp'))
        all_images = sorted(glob.glob(self.data_path + 'IMD*/IMD*_Dermoscopic_Image/*.bmp'))
        
        if len(all_images) != len(all_masks):
            raise ValueError(f"Mismatch: {len(all_images)} images but {len(all_masks)} masks")
        
        # No internal split: use all images for 'train', leave val/test empty
        if split == 'train':
            self.images = all_images
            self.masks = all_masks
        else:
            # val and test have no predefined splits for PH2
            self.images = []
            self.masks = []


    def __len__(self):
        'Returns the total number of samples'
        return len(self.images)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.images[idx]
        label_path = self.masks[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.transform:     
            # Convert to numpy arrays and ensure correct dtypes
            image_array = np.array(image)
            label_array = np.array(label)
            
            # Ensure mask is uint8 (0-255) or float (0-1) for OpenCV compatibility
            if label_array.dtype == bool:
                label_array = label_array.astype(np.uint8) * 255
            elif label_array.dtype != np.uint8 and label_array.max() <= 1.0:
                label_array = (label_array * 255).astype(np.uint8)
            elif label_array.dtype != np.uint8:
                label_array = label_array.astype(np.uint8)
            
            # Create a full-FOV ROI mask (all ones) with same spatial size as label
            roi_array = np.ones_like(label_array, dtype=np.uint8) * 255
            
            # Apply same transforms to image, lesion mask, and ROI mask
            transformed = self.transform(image=image_array, mask=label_array, mask2=roi_array) 

            X = transformed['image']
            Y = transformed['mask']
            M = transformed['mask2']
            
            # Normalize masks to [0, 1] - A.Normalize() skips masks, ToTensorV2() preserves values
            # Masks remain in [0, 255] after transform, but loss functions need [0, 1]
            if Y.max() > 1.0:
                Y = Y.float() / 255.0
            if M.max() > 1.0:
                M = M.float() / 255.0

        else: 
            X = image 
            Y = label 
            # No transforms: use a trivial full-ones ROI mask
            M = torch.ones_like(torch.as_tensor(Y, dtype=torch.float32))
            
        return X, Y, M


class DRIVEDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, path='C:/Users/ibrahim/Documents/DTU/introduction-dplrng-cv/project3/data/datasets/DRIVE/DRIVE/',
                 train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        DRIVE dataset for retinal blood vessel segmentation.
        
        Args:
            split: 'train' or 'test'
            transform: Albumentations transform or None
            path: Path to DRIVE dataset root directory
            train_ratio: (unused, kept for API compatibility)
            val_ratio: (unused, kept for API compatibility)
            seed: (unused, kept for API compatibility)
            
        Note: 
            - 'train' uses all images from DRIVE/training/ (images 21-40) with labels from '1st_manual'
            - 'test' uses all images from DRIVE/test/ (images 01-20, no labels available)
            - No internal split is performed; use external validation if needed
        """
        if transform is not None and not isinstance(transform, A.Compose):
            raise TypeError("Transform must be albumentations.Compose or None")
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        self.transform = transform
        self.data_path = os.path.normpath(path)
        self.split = split
        
        if split == 'test':
            # Test set: images 01-20 (no masks)
            images_dir = os.path.join(self.data_path, 'test', 'images')
            image_pattern = os.path.join(images_dir, '*_test.tif')
            
            self.images = sorted(glob.glob(image_pattern))
            self.masks = None
            self.fov_masks = None
        elif split == 'train':
            # Training set: use ALL images 21-40 from training/, no internal split
            images_dir = os.path.join(self.data_path, 'training', 'images')
            labels_dir = os.path.join(self.data_path, 'training', '1st_manual')
            fov_dir = os.path.join(self.data_path, 'training', 'mask')
            image_pattern = os.path.join(images_dir, '*_training.tif')
            label_pattern = os.path.join(labels_dir, '*_manual1.gif')
            fov_pattern = os.path.join(fov_dir, '*_training_mask.gif')
            
            self.images = sorted(glob.glob(image_pattern))
            self.masks = sorted(glob.glob(label_pattern))
            self.fov_masks = sorted(glob.glob(fov_pattern))
            
            if len(self.images) != len(self.masks) or len(self.images) != len(self.fov_masks):
                raise ValueError(
                    f"Mismatch: {len(self.images)} images, {len(self.masks)} labels, "
                    f"{len(self.fov_masks)} FOV masks"
                )
            
            if len(self.images) == 0:
                raise ValueError(f"No images found in dataset at: {self.data_path}. Check the path and directory structure.")
        else:
            # 'val' is not supported for DRIVE without external split
            self.images = []
            self.masks = []
            self.fov_masks = []
    
    def __len__(self):
        'Returns the total number of samples'
        return len(self.images)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.images[idx]
        image = Image.open(image_path)
        
        if self.split == 'test':
            # Test set: return only image
            if self.transform:
                transformed = self.transform(image=np.array(image))
                X = transformed['image']
            else:
                X = image
            return X
        else:
            # Train/val: return image and label from 1st_manual
            label_path = self.masks[idx]
            label = Image.open(label_path)
            fov_path = self.fov_masks[idx]
            fov = Image.open(fov_path)
            
            if self.transform:
                # Convert to numpy arrays and ensure correct dtypes
                image_array = np.array(image)
                label_array = np.array(label)
                fov_array = np.array(fov)
                
                # Ensure mask is uint8 (0-255) or float (0-1) for OpenCV compatibility
                if label_array.dtype == bool:
                    label_array = label_array.astype(np.uint8) * 255
                elif label_array.dtype != np.uint8 and label_array.max() <= 1.0:
                    label_array = (label_array * 255).astype(np.uint8)
                elif label_array.dtype != np.uint8:
                    label_array = label_array.astype(np.uint8)

                # Ensure FOV mask is uint8
                if fov_array.dtype == bool:
                    fov_array = fov_array.astype(np.uint8) * 255
                elif fov_array.dtype != np.uint8 and fov_array.max() <= 1.0:
                    fov_array = (fov_array * 255).astype(np.uint8)
                elif fov_array.dtype != np.uint8:
                    fov_array = fov_array.astype(np.uint8)
                
                # Apply same transforms to image, lesion mask, and FOV mask
                transformed = self.transform(image=image_array, mask=label_array, mask2=fov_array)
                X = transformed['image']
                Y = transformed['mask']
                M = transformed['mask2']
                
                # Normalize masks to [0, 1] - A.Normalize() skips masks, ToTensorV2() preserves values
                # Masks remain in [0, 255] after transform, but loss functions need [0, 1]
                if Y.max() > 1.0:
                    Y = Y.float() / 255.0
                if M.max() > 1.0:
                    M = M.float() / 255.0
            else:
                X = image
                Y = label
                # No transforms: use FOV mask directly as ROI mask
                M = torch.as_tensor(np.array(fov), dtype=torch.float32)
                if M.max() > 1.0:
                    M = M / 255.0
            
            return X, Y, M
