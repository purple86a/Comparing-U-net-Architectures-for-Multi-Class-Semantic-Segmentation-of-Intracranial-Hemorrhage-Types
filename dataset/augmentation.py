import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
from PIL import Image
import PIL

class HUValuePreprocessing:
    """Truncate and normalize HU values in medical images.
    
    Specifically:
    1. Truncate HU values to range [-40, 120]
    2. Normalize by subtracting 40 and dividing by 80 (maps to [-1, 1] range)
    """
    def __call__(self, image):
        """
        Args:
            image: numpy array or tensor
            
        Returns:
            Preprocessed image in same format
        """
        # Convert to numpy if it's a tensor
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image = image.numpy()
        
        # Convert to float if it's uint8 or int type
        if image.dtype == np.uint8 or np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float32)
        
        # Truncate HU values to [-40, 120]
        image = np.clip(image, -40, 120)
        
        # Normalize to [-1, 1] by subtracting 40 and dividing by 80
        image = (image - 40) / 80
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            image = torch.from_numpy(image).float()
            
        return image

class RandomRotation:
    """Rotate image and mask by a random angle."""
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, image, mask=None):
        """
        Args:
            image: PIL Image, numpy array, or torch tensor
            mask: PIL Image, numpy array, or torch tensor, optional
            
        Returns:
            Rotated image and mask in the same format as input
        """
        # Determine input type
        is_tensor = isinstance(image, torch.Tensor)
        is_pil = isinstance(image, PIL.Image.Image)
        
        # Convert input to PIL Image for rotation
        if is_tensor:
            # If tensor, convert to PIL
            if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                image_pil = F.to_pil_image(image)
            else:
                raise ValueError("Tensor must be in CHW format with 1 or 3 channels")
                
            mask_pil = None
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    mask_pil = F.to_pil_image(mask.byte())
                else:
                    raise ValueError("If image is tensor, mask should also be tensor")
        elif is_pil:
            # Already PIL
            image_pil = image
            mask_pil = mask
        else:
            # Numpy array
            if len(image.shape) == 2:  # Single channel
                image_pil = PIL.Image.fromarray(image)
            else:  # Multi-channel
                image_pil = PIL.Image.fromarray(image.astype(np.uint8))
                
            mask_pil = None
            if mask is not None:
                mask_pil = PIL.Image.fromarray(mask.astype(np.uint8))
        
        # Generate random angle
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Perform rotation
        image_rotated = F.rotate(image_pil, angle, fill=0)
        mask_rotated = F.rotate(mask_pil, angle, fill=0) if mask_pil is not None else None
        
        # Convert back to original format
        if is_tensor:
            # Convert back to tensor
            image_result = F.to_tensor(image_rotated)
            mask_result = F.to_tensor(mask_rotated).long() if mask_rotated is not None else None
        elif is_pil:
            # Keep as PIL
            image_result = image_rotated
            mask_result = mask_rotated
        else:
            # Convert back to numpy array
            image_result = np.array(image_rotated)
            mask_result = np.array(mask_rotated) if mask_rotated is not None else None
        
        return (image_result, mask_result) if mask_result is not None else image_result

class RandomCrop:
    """Crop random portion of the image and mask."""
    def __init__(self, size, padding=None):
        self.size = size
        self.padding = padding
        
    def __call__(self, image, mask=None):
        """
        Args:
            image: numpy array, tensor, or PIL Image
            mask: numpy array, tensor, or PIL Image, optional
        
        Returns:
            Cropped image and mask
        """
        # Check if the input is a PIL Image or numpy array or tensor
        is_pil = isinstance(image, PIL.Image.Image)
        is_tensor = isinstance(image, torch.Tensor)
        
        if is_pil:
            # PIL Image processing
            image_pil = image
            mask_pil = mask
        else:
            # Numpy array or tensor processing
            if is_tensor:
                image_np = image.numpy()
                mask_np = mask.numpy() if mask is not None else None
            else:
                image_np = image
                mask_np = mask
            
            # Convert to PIL for cropping
            if len(image_np.shape) == 2:  # Single channel
                image_pil = PIL.Image.fromarray(image_np.astype(np.float32))
            else:  # Multi-channel
                image_pil = PIL.Image.fromarray(image_np.astype(np.float32), mode='F')
            
            if mask_np is not None:
                if len(mask_np.shape) == 2:  # Single channel
                    mask_pil = PIL.Image.fromarray(mask_np.astype(np.uint8))
                else:  # Multi-channel
                    mask_pil = PIL.Image.fromarray(mask_np.astype(np.uint8), mode='L')
            else:
                mask_pil = None
                
        # Apply padding if specified
        if self.padding is not None:
            image_pil = F.pad(image_pil, self.padding, padding_mode='reflect')
            if mask_pil is not None:
                mask_pil = F.pad(mask_pil, self.padding, padding_mode='constant', fill=0)
        
        # Get dimensions
        w, h = F.get_image_size(image_pil)
        th, tw = self.size
        
        # Ensure crop size doesn't exceed image dimensions
        th = min(th, h)
        tw = min(tw, w)
        
        # Random top-left corner
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        # Apply crop
        image_cropped = F.crop(image_pil, i, j, th, tw)
        if mask_pil is not None:
            mask_cropped = F.crop(mask_pil, i, j, th, tw)
            
        # Convert back to original format
        if not is_pil:
            image_cropped_np = np.array(image_cropped)
            if mask_pil is not None:
                mask_cropped_np = np.array(mask_cropped)
            else:
                mask_cropped_np = None
                
            if is_tensor:
                image_result = torch.from_numpy(image_cropped_np)
                mask_result = torch.from_numpy(mask_cropped_np) if mask_cropped_np is not None else None
            else:
                image_result = image_cropped_np
                mask_result = mask_cropped_np
        else:
            image_result = image_cropped
            mask_result = mask_cropped if mask_pil is not None else None
                
        return (image_result, mask_result) if mask_result is not None else image_result

class Resize:
    """Resize the image and mask to a specified size."""
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image, mask=None):
        """
        Args:
            image: numpy array, tensor, or PIL Image
            mask: numpy array, tensor, or PIL Image, optional
        
        Returns:
            Resized image and mask
        """
        # Check if the input is a PIL Image or numpy array or tensor
        is_pil = isinstance(image, PIL.Image.Image)
        is_tensor = isinstance(image, torch.Tensor)
        
        if is_pil:
            # PIL Image processing
            image_pil = image
            mask_pil = mask
        else:
            # Numpy array or tensor processing
            if is_tensor:
                image_np = image.numpy()
                mask_np = mask.numpy() if mask is not None else None
            else:
                image_np = image
                mask_np = mask
            
            # Convert to PIL for resizing
            if len(image_np.shape) == 2:  # Single channel
                image_pil = PIL.Image.fromarray(image_np.astype(np.float32))
            else:  # Multi-channel
                image_pil = PIL.Image.fromarray(image_np.astype(np.float32), mode='F')
            
            if mask_np is not None:
                if len(mask_np.shape) == 2:  # Single channel
                    mask_pil = PIL.Image.fromarray(mask_np.astype(np.uint8))
                else:  # Multi-channel
                    mask_pil = PIL.Image.fromarray(mask_np.astype(np.uint8), mode='L')
            else:
                mask_pil = None
        
        # Apply resize
        resized_image = F.resize(image_pil, self.size, interpolation=F.InterpolationMode.BILINEAR)
        if mask_pil is not None:
            # Use nearest neighbor interpolation for masks to preserve class labels
            resized_mask = F.resize(mask_pil, self.size, interpolation=F.InterpolationMode.NEAREST)
        
        # Convert back to original format
        if not is_pil:
            resized_image_np = np.array(resized_image)
            if mask_pil is not None:
                resized_mask_np = np.array(resized_mask)
            else:
                resized_mask_np = None
                
            if is_tensor:
                image_result = torch.from_numpy(resized_image_np)
                mask_result = torch.from_numpy(resized_mask_np) if resized_mask_np is not None else None
            else:
                image_result = resized_image_np
                mask_result = resized_mask_np
        else:
            image_result = resized_image
            mask_result = resized_mask if mask_pil is not None else None
                
        return (image_result, mask_result) if mask_result is not None else image_result

class ToTensor:
    """Convert image and mask to PyTorch tensors."""
    def __call__(self, image, mask=None):
        """
        Args:
            image: numpy array or PIL Image
            mask: numpy array or PIL Image, optional
        
        Returns:
            Tensor image and mask
        """
        # Check if input is already a tensor
        if isinstance(image, torch.Tensor):
            img_tensor = image
            mask_tensor = mask
            return (img_tensor, mask_tensor) if mask is not None else img_tensor
            
        # Check if the input is a PIL Image
        is_pil = isinstance(image, PIL.Image.Image)
        
        if is_pil:
            # Convert PIL Image to numpy array
            image_np = np.array(image).astype(np.float32)
            mask_np = np.array(mask).astype(np.uint8) if mask is not None else None
        else:
            # Input is already numpy array
            image_np = image.astype(np.float32)
            mask_np = mask.astype(np.uint8) if mask is not None else None
        
        # Handle different dimensions
        if len(image_np.shape) == 2:  # Single channel
            image_np = image_np[..., np.newaxis]
        
        # Convert to tensor
        # For image: normalize to [0,1] if not already and move channel to first dimension
        if image_np.max() > 1.0:
            img_tensor = torch.from_numpy(image_np / 255.0).permute(2, 0, 1).float()
        else:
            img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        if mask_np is not None:
            if len(mask_np.shape) == 2:  # Single channel
                mask_tensor = torch.from_numpy(mask_np).long()
            else:  # Multi-channel
                mask_tensor = torch.from_numpy(mask_np).permute(2, 0, 1).long()
            return img_tensor, mask_tensor
        else:
            return img_tensor

class AugmentationPipeline:
    """Full augmentation pipeline for medical image segmentation.
    
    Includes:
    1. HU value truncation and normalization
    2. Spatial augmentations (rotation, resize, crop)
    """
    def __init__(self, output_size=(512, 512), train=True):
        self.train = train
        self.output_size = output_size
        
        # HU preprocessing always applied
        self.hu_preprocessing = HUValuePreprocessing()
        
        # Training augmentations
        self.train_transforms = [
            RandomRotation(degrees=15),
            RandomCrop(size=(int(output_size[0]*0.9), int(output_size[1]*0.9))),
            Resize(output_size)
        ]
        
    def __call__(self, image, mask=None):
        """
        Apply transformations to image and mask
        
        Args:
            image: PIL Image or numpy array
            mask: PIL Image or numpy array (optional)
            
        Returns:
            Transformed image and mask
        """
        # Apply HU preprocessing to image only (works on numpy array)
        image = self.hu_preprocessing(image)
        
        # Convert normalized array back to uint8 range for PIL operations
        # Scale from [-1, 1] to [0, 255]
        image_uint8 = ((image + 1) * 127.5).astype(np.uint8)
        
        # Keep mask as is, just ensure it's uint8
        if mask is not None:
            mask_uint8 = mask.astype(np.uint8)
        
        # Apply spatial augmentations to both image and mask during training
        if self.train:
            for transform in self.train_transforms:
                if mask is not None:
                    image_uint8, mask_uint8 = transform(image_uint8, mask_uint8)
                else:
                    image_uint8 = transform(image_uint8)
                    
        # Always ensure output size even for validation/test
        if not self.train:
            resize = Resize(self.output_size)
            if mask is not None:
                image_uint8, mask_uint8 = resize(image_uint8, mask_uint8)
            else:
                image_uint8 = resize(image_uint8)
        
        # Convert PIL Image back to tensor
        if isinstance(image_uint8, Image.Image):
            image_tensor = F.to_tensor(image_uint8)  # Range [0, 1]
            # Rescale back to [-1, 1]
            image_tensor = image_tensor * 2 - 1
        else:
            # Just in case we're not working with PIL Image
            image_tensor = torch.tensor(image_uint8, dtype=torch.float32) / 127.5 - 1
            image_tensor = image_tensor.unsqueeze(0) if len(image_tensor.shape) == 2 else image_tensor.permute(2, 0, 1)
        
        # Convert mask to tensor
        if mask is not None:
            if isinstance(mask_uint8, Image.Image):
                mask_tensor = torch.tensor(np.array(mask_uint8), dtype=torch.long)
            else:
                mask_tensor = torch.tensor(mask_uint8, dtype=torch.long)
                
            return image_tensor, mask_tensor
        else:
            return image_tensor

def get_train_transform(output_size=(512, 512)):
    """Get augmentation pipeline for training data"""
    return AugmentationPipeline(output_size=output_size, train=True)
    
def get_val_transform(output_size=(512, 512)):
    """Get augmentation pipeline for validation/test data"""
    return AugmentationPipeline(output_size=output_size, train=False)

# Example of how to integrate with your dataset class
class AugmentedSliceDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform or get_val_transform()  # Default transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx])
        img_array = np.array(img)
        
        # Load mask
        mask = Image.open(self.mask_paths[idx])
        mask_array = np.array(mask)
        
        # Apply transforms (including HU preprocessing and augmentations)
        if self.transform:
            img_tensor, mask_tensor = self.transform(img_array, mask_array)
        else:
            # Convert to tensor if no transform
            img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask_array, dtype=torch.long)
        
        return img_tensor, mask_tensor 