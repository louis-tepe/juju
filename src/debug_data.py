import os
import tensorflow as tf
import numpy as np
import cv2
import hydra
from src.data.loader import get_dataset
from src.utils.seeding import seed_everything

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    seed_everything(cfg.seed)
    
    # Get Train Dataset
    ds = get_dataset(cfg, split='train')
    
    print("Iterating through dataset...")
    for i, (images, labels) in enumerate(ds.take(1)):
        print(f"Batch {i} shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Images range: [{np.min(images)}, {np.max(images)}]")
        print(f"Images mean: {np.mean(images)}, std: {np.std(images)}")
        print(f"Labels sample: {labels[0]}")
        
        # Save first 5 images
        for j in range(5):
            img = images[j].numpy()
            # Denormalize if needed (loader divides by 255)
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"debug_img_{j}.png", img)
            print(f"Saved debug_img_{j}.png")

if __name__ == "__main__":
    main()
