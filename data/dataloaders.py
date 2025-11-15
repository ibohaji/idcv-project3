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
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducible splits
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
        
        # Create splits with fixed seed for reproducibility
        np.random.seed(seed)
        indices = np.random.permutation(len(all_images))
        
        n_total = len(all_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Test gets the remainder
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Select indices based on split
        if split == 'train':
            split_indices = train_indices
        elif split == 'val':
            split_indices = val_indices
        else:  # test
            split_indices = test_indices
        
        self.images = [all_images[i] for i in split_indices]
        self.masks = [all_masks[i] for i in split_indices]


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
            
            transformed = self.transform(image=image_array, mask=label_array) 

            X = transformed['image']
            Y = transformed['mask']
            
            # Normalize mask to [0, 1] - A.Normalize() skips masks, ToTensorV2() preserves values
            # Masks remain in [0, 255] after transform, but loss functions need [0, 1]
            if Y.max() > 1.0:
                Y = Y.float() / 255.0

        else: 
            X = image 
            Y = label 
            
        return X, Y


class DRIVEDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, path='C:/Users/ibrahim/Documents/DTU/introduction-dplrng-cv/project3/data/datasets/DRIVE/DRIVE/',
                 train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        DRIVE dataset for retinal blood vessel segmentation.
        
        Args:
            split: 'train', 'val', or 'test'
            transform: Albumentations transform or None
            path: Path to DRIVE dataset root directory
            train_ratio: Proportion of training data for train split (rest goes to val)
            val_ratio: Proportion of training data for val split
            seed: Random seed for reproducible splits
            
        Note: 
            - Training/val splits are created from the original training set (images 21-40)
            - Test set returns only images (no labels)
            - Training/val use labels from '1st_manual' folder
        """
        if transform is not None and not isinstance(transform, A.Compose):
            raise TypeError("Transform must be albumentations.Compose or None")
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        self.transform = transform
        self.data_path = os.path.normpath(path)  # Normalize path for cross-platform compatibility
        self.split = split
        
        if split == 'test':
            # Test set: images 01-20 (no masks)
            images_dir = os.path.join(self.data_path, 'test', 'images')
            image_pattern = os.path.join(images_dir, '*_test.tif')
            
            self.images = sorted(glob.glob(image_pattern))
            self.masks = None
        else:
            # Training set: images 21-40, use 1st_manual as labels
            images_dir = os.path.join(self.data_path, 'training', 'images')
            labels_dir = os.path.join(self.data_path, 'training', '1st_manual')
            image_pattern = os.path.join(images_dir, '*_training.tif')
            label_pattern = os.path.join(labels_dir, '*_manual1.gif')
            
            all_images = sorted(glob.glob(image_pattern))
            all_labels = sorted(glob.glob(label_pattern))
            
            if len(all_images) != len(all_labels):
                raise ValueError(f"Mismatch: {len(all_images)} images but {len(all_labels)} labels")
            
            if len(all_images) == 0:
                raise ValueError(f"No images found in dataset at: {self.data_path}. Check the path and directory structure.")
            
            # Split training data into train/val
            np.random.seed(seed)
            indices = np.random.permutation(len(all_images))
            
            n_total = len(all_images)
            n_train = int(n_total * train_ratio)
            # Val gets the remainder
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            if split == 'train':
                split_indices = train_indices
            else:  # val
                split_indices = val_indices
            
            self.images = [all_images[i] for i in split_indices]
            self.masks = [all_labels[i] for i in split_indices]
        
        if self.split != 'test' and len(self.images) != len(self.masks):
            raise ValueError(f"Mismatch: {len(self.images)} images but {len(self.masks)} labels")
    
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
                
                transformed = self.transform(image=image_array, mask=label_array)
                X = transformed['image']
                Y = transformed['mask']
                
                # Normalize mask to [0, 1] - A.Normalize() skips masks, ToTensorV2() preserves values
                # Masks remain in [0, 255] after transform, but loss functions need [0, 1]
                if Y.max() > 1.0:
                    Y = Y.float() / 255.0
            else:
                X = image
                Y = label
            
            return X, Y
