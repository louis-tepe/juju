import os
from functools import partial

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.preprocess import load_and_preprocess_image


class DataLoader:
    """Optimized DataLoader for M2 unified memory architecture."""
    
    def __init__(self, config):
        self.config = config
        self.autotune = tf.data.AUTOTUNE

    def _load_image(self, path, label):
        """Load and resize image WITHOUT normalization (for augmentation)."""
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [self.config.data.image_size, self.config.data.image_size])
        # Keep as float32 [0, 255] for augmentation
        return img, label

    def _augment_image_native(self, image, label):
        """Safe geometric augmentations for medical/retinal images."""
        # Geometric transforms only - safe for medical imaging
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        return image, label

    def _normalize_image(self, image, label):
        """Apply EfficientNet normalization AFTER augmentation."""
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label

    def build_dataset(self, df: pd.DataFrame, training: bool = True):
        """Builds optimized tf.data.Dataset for M2."""
        paths = df['path'].values
        labels = df['diagnosis'].values.astype(np.int32)
        labels = tf.one_hot(labels, depth=5)

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        # Shuffle before mapping for training
        if training:
            dataset = dataset.shuffle(buffer_size=min(1000, len(paths)), reshuffle_each_iteration=True)

        # Load images (no normalization yet)
        dataset = dataset.map(self._load_image, num_parallel_calls=self.autotune)
        
        # Cache raw images (before augmentation for variety)
        use_cache = getattr(self.config.data, 'use_cache', True)
        if use_cache and self.config.data.image_size <= 256:
            dataset = dataset.cache()
        
        # Augmentation on raw [0, 255] images (training only)
        if training:
            dataset = dataset.map(self._augment_image_native, num_parallel_calls=self.autotune)

        # Normalize AFTER augmentation
        dataset = dataset.map(self._normalize_image, num_parallel_calls=self.autotune)

        # Batch and prefetch
        dataset = dataset.batch(self.config.data.batch_size)
        dataset = dataset.prefetch(self.autotune)

        return dataset


def get_dataset(config, split='train'):
    """Factory function to get dataset based on split."""
    base_dir = hydra.utils.to_absolute_path(
        config.data.train_images if split in ['train', 'val'] else config.data.test_images
    )

    if split in ['train', 'val']:
        csv_path = hydra.utils.to_absolute_path(config.data.train_folds_csv)
        if not os.path.exists(csv_path):
            print(f"Folds file {csv_path} not found, using raw {config.data.train_csv}")
            csv_path = hydra.utils.to_absolute_path(config.data.train_csv)
            df = pd.read_csv(csv_path)
            if 'fold' not in df.columns:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
                df['fold'] = -1
                for fold, (t, v) in enumerate(skf.split(df, df['diagnosis'])):
                    df.loc[v, 'fold'] = fold
        else:
            df = pd.read_csv(csv_path)

        fold = config.train.fold
        if split == 'train':
            df = df[df['fold'] != fold]
        elif split == 'val':
            df = df[df['fold'] == fold]
    else:
        csv_path = hydra.utils.to_absolute_path(config.data.test_csv)
        if not os.path.exists(csv_path):
            df = pd.DataFrame({'id_code': ['0005cfc8afb6']})
        else:
            df = pd.read_csv(csv_path)

    df['path'] = df['id_code'].apply(lambda x: os.path.join(base_dir, f"{x}.png"))

    if split == 'train':
        return DataLoader(config).build_dataset(df, training=True)
    elif split == 'val':
        return DataLoader(config).build_dataset(df, training=False)
    else:
        if 'diagnosis' not in df.columns:
            df['diagnosis'] = 0
        return DataLoader(config).build_dataset(df, training=False)
