import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image

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

# Create output directory if it doesn't exist
os.makedirs('data/images', exist_ok=True)

# Directory containing the 3D volumes
input_dir = 'MBH_train_label/imagesTr'

# Get all .nii.gz files in the input directory
files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]

print(f"Found {len(files)} files to process")

# Process each file
for file in tqdm(files):
    file_path = os.path.join(input_dir, file)
    try:
        # Read the 3D volume
        image = sitk.ReadImage(file_path)
        volume = sitk.GetArrayFromImage(image)  # Shape: (Z, Y, X)
        
        # Get the base filename without extension
        basename = os.path.splitext(os.path.splitext(file)[0])[0]
        
        # Save each slice as a 2D image
        for z in range(volume.shape[0]):
            slice_2d = volume[z]
            
            # Convert to PNG and save
            normalized_image = normalize_for_png(slice_2d)
            img = Image.fromarray(normalized_image)
            output_path = f'data/images/{basename}_slice_{z:03d}.png'
            img.save(output_path)
        
        # print(f"Processed {file}: {volume.shape[0]} slices extracted")
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

print("All files processed successfully!") 