import time
import numpy as np
import albumentations as A
import cv2

def get_heavy_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
    ])

def get_light_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        # Removed Distortions
    ])

def benchmark(name, aug, num_images=100, size=512):
    print(f"Benchmarking {name}...")
    image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    
    start = time.time()
    for _ in range(num_images):
        _ = aug(image=image)['image']
    end = time.time()
    
    duration = end - start
    fps = num_images / duration
    print(f"  Processed {num_images} images in {duration:.4f}s")
    print(f"  Speed: {fps:.2f} images/sec")
    return fps

if __name__ == "__main__":
    print("Benchmarking Albumentations on CPU")
    print("-" * 30)
    
    heavy_fps = benchmark("Heavy Augmentations", get_heavy_augmentations())
    light_fps = benchmark("Light Augmentations", get_light_augmentations())
    
    print("-" * 30)
    print(f"Speedup with Light Augs: {light_fps / heavy_fps:.2f}x")
