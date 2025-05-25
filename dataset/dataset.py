import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms
import multiprocessing

class SliceDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
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
        
        # Apply transforms if specified
        if self.transform:
            img_array = self.transform(img_array)
        else:
            # Convert to tensor and normalize
            img_array = img_array.astype(np.float32) / 255.0
            img_array = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # Add channel dim: (1, H, W)
        
        # Convert mask to tensor (no normalization for masks)
        mask_tensor = torch.tensor(mask_array, dtype=torch.long)  # (H, W)
        
        return img_array, mask_tensor

def get_dataset_paths():
    """Get all paired image and mask paths"""
    image_dir = 'data/images'
    mask_dir = 'data/masks'
    
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

def create_data_loaders(batch_size=16, val_split=0.15, test_split=0.15, seed=42, num_workers=0):
    """Create train, validation and test data loaders"""
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
    
    # Create datasets
    train_dataset = SliceDataset(train_imgs, train_masks)
    val_dataset = SliceDataset(val_imgs, val_masks)
    test_dataset = SliceDataset(test_imgs, test_masks)
    
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
    # This is required on Windows for multiprocessing
    multiprocessing.freeze_support()
    
    # Example usage
    image_paths, mask_paths = get_dataset_paths()
    
    # Calculate class weights
    class_weights = get_class_weights(mask_paths)
    print(f"Class weights: {class_weights}")
    
    # Create data loaders with 0 workers for testing
    # Increase num_workers only in production
    train_loader, val_loader, test_loader = create_data_loaders(num_workers=0)
    
    # Inspect a few samples
    for images, masks in train_loader:
        print(f"Batch shape: {images.shape}, {masks.shape}")
        print(f"Unique mask values: {torch.unique(masks).numpy()}")
        
        # Check class distribution in this batch
        for c in range(6):
            pixel_count = torch.sum(masks == c).item()
            if pixel_count > 0:
                print(f"Class {c}: {pixel_count} pixels")
        break 