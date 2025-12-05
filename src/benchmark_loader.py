import time
import tensorflow as tf
import hydra
from omegaconf import DictConfig
from src.data.loader import get_dataset
import os

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(f"Benchmarking Data Loader with Batch Size: {cfg.data.batch_size}")
    print(f"Image Size: {cfg.data.image_size}")
    print(f"Augmentations: Enabled (if training=True)")
    
    # Force training split to enable augmentations
    ds = get_dataset(cfg, split='train')
    
    # Warmup
    print("Warming up...")
    for _ in ds.take(5):
        pass
        
    print("Starting Benchmark...")
    start_time = time.time()
    count = 0
    num_batches = 100
    
    for _ in ds.take(num_batches):
        count += 1
        
    end_time = time.time()
    duration = end_time - start_time
    images_processed = count * cfg.data.batch_size
    
    print(f"Processed {count} batches ({images_processed} images) in {duration:.2f} seconds.")
    print(f"Throughput: {images_processed / duration:.2f} images/sec")
    print(f"Latency: {duration / count * 1000:.2f} ms/batch")

if __name__ == "__main__":
    main()
