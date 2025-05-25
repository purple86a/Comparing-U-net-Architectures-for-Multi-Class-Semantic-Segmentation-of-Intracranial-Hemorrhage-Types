import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from sklearn.model_selection import train_test_split

from augmentation import get_train_transform, get_val_transform

class AugmentedSliceDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        
        # Use the appropriate transform based on the dataset split
        if transform is None:
            self.transform = get_train_transform() if is_train else get_val_transform()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx])
        img_array = np.array(img)
        
        # Load mask
        mask = Image.open(self.mask_paths[idx])
        mask_array = np.array(mask)
        
        # Apply all transforms (HU preprocessing + augmentations if training)
        img_tensor, mask_tensor = self.transform(img_array, mask_array)
        
        return img_tensor, mask_tensor


class SimpleAugmentedDataset(Dataset):
    """Dataset class that simply loads pre-augmented data without applying transformations"""
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx])
        img_array = np.array(img, dtype=np.float32)
        
        # Load mask
        mask = Image.open(self.mask_paths[idx])
        mask_array = np.array(mask, dtype=np.int64)
        
        # Convert to tensors directly without additional transforms
        # Normalize the image to [-1, 1] range (assuming HU values are already processed in saved images)
        img_array = img_array / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()  # Add channel dimension
        mask_tensor = torch.from_numpy(mask_array).long()
        
        return img_tensor, mask_tensor


def get_dataset_paths():
    """Get all paired image and mask paths"""
    image_dir = 'augmented_data/images'
    mask_dir = 'augmented_data/masks'
    
    # Get all PNG files in the image directory
    all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    all_images.sort()  # Ensure consistent ordering
    
    # Create corresponding mask paths
    all_masks = []
    valid_images = []
    
    for img_path in all_images:
        # Get the corresponding mask path
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, filename)
        
        # Only include if both image and mask exist
        if os.path.exists(mask_path):
            valid_images.append(img_path)
            all_masks.append(mask_path)
    
    print(f"Found {len(valid_images)} valid image-mask pairs")
    return valid_images, all_masks

def create_augmented_data_loaders(batch_size=16, val_split=0.15, test_split=0.15, seed=42, num_workers=0):
    """Create train, validation and test data loaders with augmentation"""
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get image and mask paths
    image_paths, mask_paths = get_dataset_paths()
    
    # Split into train, validation, and test sets
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=test_split, random_state=seed
    )
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_imgs, train_masks, test_size=val_split/(1-test_split), random_state=seed
    )
    
    print(f"Training: {len(train_imgs)} samples")
    print(f"Validation: {len(val_imgs)} samples")
    print(f"Testing: {len(test_imgs)} samples")
    
    # Create augmented datasets with appropriate transforms
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = AugmentedSliceDataset(train_imgs, train_masks, transform=train_transform, is_train=True)
    val_dataset = AugmentedSliceDataset(val_imgs, val_masks, transform=val_transform, is_train=False)
    test_dataset = AugmentedSliceDataset(test_imgs, test_masks, transform=val_transform, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def create_simple_data_loaders(batch_size=16, val_split=0.15, test_split=0.15, seed=42, num_workers=0):
    """Create train, validation and test data loaders with NO augmentation - using pre-saved augmented data"""
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get image and mask paths
    image_paths, mask_paths = get_dataset_paths()
    
    # Split into train, validation, and test sets
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=test_split, random_state=seed
    )
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_imgs, train_masks, test_size=val_split/(1-test_split), random_state=seed
    )
    
    print(f"Training: {len(train_imgs)} samples")
    print(f"Validation: {len(val_imgs)} samples")
    print(f"Testing: {len(test_imgs)} samples")
    
    # Create simple datasets without transforms
    train_dataset = SimpleAugmentedDataset(train_imgs, train_masks)
    val_dataset = SimpleAugmentedDataset(val_imgs, val_masks)
    test_dataset = SimpleAugmentedDataset(test_imgs, test_masks)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def get_class_weights(mask_paths, num_classes=6):
    """Calculate class weights to handle class imbalance"""
    class_counts = np.zeros(num_classes)
    
    print("Calculating class distribution...")
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        for c in range(num_classes):
            class_counts[c] += np.sum(mask == c)
    
    # Calculate weights (inverse frequency)
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (class_counts * num_classes)
    
    # Replace infinities with a large number
    class_weights = np.nan_to_num(class_weights, nan=1.0, posinf=10.0, neginf=1.0)
    
    return torch.FloatTensor(class_weights)

if __name__ == "__main__":
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Example usage
    image_paths, mask_paths = get_dataset_paths()
    
    # Calculate class weights
    class_weights = get_class_weights(mask_paths)
    print(f"Class weights: {class_weights}")
    
    # Create data loaders with 0 workers for testing
    # Increase num_workers only in production
    train_loader, val_loader, test_loader = create_augmented_data_loaders(num_workers=0)
    
    # Inspect a few samples
    for images, masks in train_loader:
        print(f"Batch shape: {images.shape}, {masks.shape}")
        print(f"Image value range: [{images.min().item():.2f}, {images.max().item():.2f}]")
        print(f"Unique mask values: {torch.unique(masks).numpy()}")
        
        # Check class distribution in this batch
        for c in range(6):
            pixel_count = torch.sum(masks == c).item()
            if pixel_count > 0:
                print(f"Class {c}: {pixel_count} pixels")
        break 