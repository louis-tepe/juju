"""
Augmentation pipelines for APTOS 2019 Diabetic Retinopathy Detection.

Based on top solutions which used Albumentations for strong regularization
while being careful not to destroy retinal lesion patterns.
"""
import albumentations as A
import numpy as np


def get_train_transforms(image_size: int = 380):
    """
    Strong but safe augmentations for retinal images.
    
    Safe transforms:
    - Geometric: flip, rotate, shift/scale (retina can be in any orientation)
    - Minor blur (simulates camera focus variance)
    - Brightness/contrast (lighting conditions vary between clinics)
    - CLAHE (enhances local contrast, useful for retinal imaging)
    - CoarseDropout (regularization, simulates artifacts/occlusions)
    
    Avoided transforms:
    - Extreme color shifts (could mask hemorrhages)
    - Elastic distortion (distorts lesion shapes)
    - Heavy noise (destroys fine details)
    """
    return A.Compose([
        # Geometric transforms (safe - retina has no fixed orientation)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,  # Reduced from 0.1
            scale_limit=0.1,   # Reduced from 0.15
            rotate_limit=10,   # Reduced from 15
            border_mode=0,     # Black border
            p=0.3              # Reduced from 0.5
        ),
        
        # REMOVED: GridDistortion and OpticalDistortion
        # These can destroy lesion patterns and confuse the model
        
        # Color/brightness (conservative - preserve diagnostic features)
        A.RandomBrightnessContrast(
            brightness_limit=0.1,  # Reduced from 0.2
            contrast_limit=0.1,    # Reduced from 0.2
            p=0.3                  # Reduced from 0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,     # Reduced from 10
            sat_shift_limit=10,    # Reduced from 20
            val_shift_limit=5,     # Reduced from 10
            p=0.2                  # Reduced from 0.3
        ),
        
        # REMOVED: RandomGamma (can alter lesion visibility)
        
        # Blur (simulates camera focus variance) - kept light
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3),
        ], p=0.2),  # Reduced from 0.3
        
        # CLAHE for local contrast enhancement (beneficial for retinas)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),  # Reduced from 0.3
        
        # Light regularization dropout
        A.CoarseDropout(
            max_holes=4,       # Reduced from 6
            max_height=24,     # Reduced from 32
            max_width=24,      # Reduced from 32
            min_holes=1,       # Reduced from 2
            min_height=8,
            min_width=8,
            fill=0,
            p=0.15             # Reduced from 0.3
        ),
    ])


def get_val_transforms(image_size: int = 380):
    """No augmentation for validation - just return identity."""
    return A.Compose([])


def apply_transforms(image: np.ndarray, transforms: A.Compose) -> np.ndarray:
    """Apply albumentations transforms to an image."""
    if transforms is not None:
        augmented = transforms(image=image)
        return augmented['image']
    return image
