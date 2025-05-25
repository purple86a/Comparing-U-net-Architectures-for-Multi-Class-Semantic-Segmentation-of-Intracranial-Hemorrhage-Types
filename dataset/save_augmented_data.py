import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

from augmentation import get_train_transform
from dataset_augmented import get_dataset_paths

def save_augmented_data(num_augmentations=5, seed=42):
    """
    Generate and save augmented data
    
    Args:
        num_augmentations: Number of augmentations to generate per original image
        seed: Random seed for reproducibility
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get image and mask paths
    print("Loading dataset paths...")
    image_paths, mask_paths = get_dataset_paths()
    
    # Create output directories if they don't exist
    os.makedirs('augmented_data/images', exist_ok=True)
    os.makedirs('augmented_data/masks', exist_ok=True)
    
    # Get the train transform for augmentation
    transform = get_train_transform()
    
    # Track how many augmented images we've created
    augmented_count = 0
    
    # First, copy the original images to augmented_data
    print(f"Copying original images and masks...")
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load original image and mask
        orig_img = Image.open(img_path)
        orig_mask = Image.open(mask_path)
        
        # Get base filename
        base_filename = os.path.basename(img_path)
        name, ext = os.path.splitext(base_filename)
        
        # Save original image and mask
        orig_img.save(f'augmented_data/images/{name}_orig{ext}')
        orig_mask.save(f'augmented_data/masks/{name}_orig{ext}')
        augmented_count += 1
    
    # Now generate augmented versions
    print(f"Generating {num_augmentations} augmented versions per original image...")
    for i, (img_path, mask_path) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths))):
        # Load image and mask as arrays
        img = Image.open(img_path)
        img_array = np.array(img)
        
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        # Get base filename
        base_filename = os.path.basename(img_path)
        name, ext = os.path.splitext(base_filename)
        
        # Generate multiple augmentations per image
        for aug_idx in range(num_augmentations):
            # Apply augmentation
            aug_img_tensor, aug_mask_tensor = transform(img_array, mask_array)
            
            # Convert tensors back to PIL images
            # Handle different tensor formats
            if aug_img_tensor.shape[0] == 1:  # Single channel
                # Convert to grayscale image
                aug_img_array = ((aug_img_tensor[0].numpy() + 1) * 127.5).astype(np.uint8)
                aug_img = Image.fromarray(aug_img_array, mode='L')
            else:
                # Convert to RGB image
                aug_img_array = ((aug_img_tensor.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
                # If shape is (H, W, 1), squeeze the channel dimension
                if aug_img_array.shape[2] == 1:
                    aug_img_array = aug_img_array.squeeze(2)
                    aug_img = Image.fromarray(aug_img_array, mode='L')
                else:
                    aug_img = Image.fromarray(aug_img_array, mode='RGB')
            
            # Handle mask tensor
            aug_mask_array = aug_mask_tensor.numpy().astype(np.uint8)
            aug_mask = Image.fromarray(aug_mask_array, mode='L')
            
            # Save augmented image and mask
            aug_img.save(f'augmented_data/images/{name}_aug{aug_idx}{ext}')
            aug_mask.save(f'augmented_data/masks/{name}_aug{aug_idx}{ext}')
            augmented_count += 1
    
    print(f"Saved {augmented_count} total images and masks to augmented_data directory")
    print(f"  - Original images: {len(image_paths)}")
    print(f"  - Augmented images: {len(image_paths) * num_augmentations}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate and save augmented data')
    parser.add_argument('--num_augmentations', type=int, default=5, 
                        help='Number of augmentations to generate per original image')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Generate and save augmented data
    save_augmented_data(num_augmentations=args.num_augmentations, seed=args.seed) 