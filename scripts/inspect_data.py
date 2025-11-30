import hydra
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import get_dataset
import os

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg):
    print("Inspecting Data Loading...")
    
    # Get Train Dataset
    ds = get_dataset(cfg, split='train')
    
    # Take one batch
    for images, labels in ds.take(1):
        print(f"Batch Shape: {images.shape}")
        print(f"Labels Shape: {labels.shape}")
        
        print(f"Image Min: {np.min(images)}")
        print(f"Image Max: {np.max(images)}")
        print(f"Image Mean: {np.mean(images)}")
        
        print(f"Labels Sample:\n{labels[:5]}")
        
        # Save a few images
        for i in range(5):
            img = images[i].numpy()
            # If Ben Graham, values might be around 128. 
            # If standard, 0-255.
            # Normalize for display if needed, but let's save raw to see.
            
            # Cast to uint8 for saving
            img_save = np.clip(img, 0, 255).astype(np.uint8)
            
            import cv2
            cv2.imwrite(f"debug_img_{i}.png", cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
            
        break

if __name__ == "__main__":
    main()
