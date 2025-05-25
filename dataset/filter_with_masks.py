import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import shutil
from PIL import Image

# Create output directories if they don't exist
os.makedirs('data/images', exist_ok=True)
os.makedirs('data/masks', exist_ok=True)

# Directories containing the 3D volumes and masks
image_dir = 'MBH_train_label/imagesTr'
mask_dir = 'MBH_train_label/labelsTr'

# Get all .nii.gz files in the input directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]

print(f"Found {len(image_files)} files to process")

# Hemorrhage class encodings
# 0: Background (no hemorrhage)
# 1: EDH (Epidural Hemorrhage)
# 2: IPH (Intraparenchymal Hemorrhage)
# 3: IVH (Intraventricular Hemorrhage)
# 4: SAH (Subarachnoid Hemorrhage)
# 5: SDH (Subdural Hemorrhage)

# Set random seed for reproducibility
np.random.seed(42)

# Function to normalize image for PNG format
def normalize_for_png(image_slice):
    # Normalize to 0-255 range
    img_min = np.min(image_slice)
    img_max = np.max(image_slice)
    
    if img_max > img_min:
        normalized = ((image_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image_slice, dtype=np.uint8)
    
    return normalized

# Function to check if mask has any hemorrhage classes (values 1-5)
def has_hemorrhage(mask_slice):
    # Check if the mask slice contains any voxels with value 1, 2, 3, 4, or 5
    for class_id in range(1, 6):
        if np.any(mask_slice == class_id):
            return True
    return False

# Process each file
for file in tqdm(image_files):
    image_path = os.path.join(image_dir, file)
    mask_path = os.path.join(mask_dir, file)
    
    # Skip if mask file doesn't exist
    if not os.path.exists(mask_path):
        print(f"Warning: Mask file {mask_path} not found, skipping {file}")
        continue
    
    try:
        # Read the 3D volume and mask
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # Convert to numpy arrays
        image_volume = sitk.GetArrayFromImage(image)  # Shape: (Z, Y, X)
        mask_volume = sitk.GetArrayFromImage(mask)    # Shape: (Z, Y, X)
        
        # Get the base filename without extension
        basename = os.path.splitext(os.path.splitext(file)[0])[0]
        
        # Process each slice
        for z in range(image_volume.shape[0]):
            image_slice = image_volume[z]
            mask_slice = mask_volume[z]
            
            slice_basename = f'{basename}_slice_{z:03d}'
            image_output_path = f'data/images/{slice_basename}.png'
            mask_output_path = f'data/masks/{slice_basename}.png'
            
            # Check if mask has any hemorrhage classes or if we should keep it randomly
            if has_hemorrhage(mask_slice) or np.random.rand() < 0.01:
                # Normalize and save the image slice as PNG
                normalized_image = normalize_for_png(image_slice)
                img = Image.fromarray(normalized_image)
                img.save(image_output_path)
                
                # Save mask as PNG with class labels preserved
                # Mask values 0-5 are well within PNG's 8-bit range (0-255)
                mask_img = Image.fromarray(mask_slice.astype(np.uint8))
                mask_img.save(mask_output_path)
                
                # Optional: Print class distribution in this slice
                class_counts = {class_id: np.sum(mask_slice == class_id) for class_id in range(1, 6)}
                if has_hemorrhage(mask_slice):
                    present_classes = [f"Class {class_id}" for class_id in range(1, 6) if np.any(mask_slice == class_id)]
                    # print(f"Slice {slice_basename} contains: {', '.join(present_classes)}")
            else:
                # Delete the image file if it exists and mask has no hemorrhage
                if os.path.exists(image_output_path):
                    os.remove(image_output_path)
        
        # print(f"Processed {file}")
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

print("All files processed successfully!") 