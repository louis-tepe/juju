import os
from functools import partial

import albumentations as A
import cv2
import hydra
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.preprocess import load_and_preprocess_image, crop_image_from_gray
from src.data.augmentations import get_train_transforms, get_val_transforms, apply_transforms


class DataLoader:
    """Optimized DataLoader for M2 unified memory architecture."""
    
    def __init__(self, config):
        self.config = config
        self.autotune = tf.data.AUTOTUNE
        self.use_ben_graham = getattr(config.data, 'use_ben_graham', False)
        self.use_mixup = getattr(config.data, 'use_mixup', False)
        self.mixup_alpha = getattr(config.data, 'mixup_alpha', 0.4)
        
        # Use strong Albumentations-based augmentations (recommended for QWK)
        self.use_strong_augment = getattr(config.data, 'use_strong_augment', True)
        
        # Detect regression mode from model config
        self.is_regression = getattr(config.model, 'output_type', 'softmax') == 'regression'
        
        # Disable Mixup for regression (label mixing doesn't work with scalar targets)
        if self.is_regression and self.use_mixup:
            print("⚠️ Mixup disabled for regression mode (incompatible with scalar labels)")
            self.use_mixup = False
        
        # Initialize Albumentations transforms
        if self.use_strong_augment:
            self.train_transforms = get_train_transforms(config.data.image_size)
            self.val_transforms = get_val_transforms(config.data.image_size)
            print("✅ Using strong Albumentations augmentations")

    def _load_image(self, path, label):
        """Load and resize image WITHOUT normalization (for augmentation)."""
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [self.config.data.image_size, self.config.data.image_size])
        # Keep as float32 [0, 255] for augmentation
        return img, label

    def _apply_ben_graham_np(self, image):
        """Ben Graham preprocessing: crop + Gaussian normalize contrast."""
        image = image.numpy().astype(np.uint8)
        # Crop black borders
        image = crop_image_from_gray(image)
        # Resize after crop
        image = cv2.resize(image, (self.config.data.image_size, self.config.data.image_size))
        # Ben Graham formula: 4*img - 4*blur + 128
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
        return image.astype(np.float32)

    def _apply_ben_graham(self, image, label):
        """TF wrapper for Ben Graham preprocessing."""
        image = tf.py_function(
            self._apply_ben_graham_np,
            [image],
            tf.float32
        )
        image.set_shape([self.config.data.image_size, self.config.data.image_size, 3])
        return image, label

    def _augment_image_native(self, image, label):
        """Safe geometric augmentations for medical/retinal images (basic TF ops)."""
        # Geometric transforms only - safe for medical imaging
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        return image, label

    def _augment_image_albumentations_np(self, image):
        """Apply Albumentations transforms (numpy function)."""
        image = image.numpy().astype(np.uint8)
        augmented = self.train_transforms(image=image)
        return augmented['image'].astype(np.float32)

    def _augment_image_albumentations(self, image, label):
        """TF wrapper for Albumentations augmentations."""
        image = tf.py_function(
            self._augment_image_albumentations_np,
            [image],
            tf.float32
        )
        image.set_shape([self.config.data.image_size, self.config.data.image_size, 3])
        return image, label

    def _normalize_image(self, image, label):
        """Apply EfficientNet normalization AFTER augmentation."""
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label

    def _mixup_batch(self, images, labels):
        """Mixup augmentation: λ*x1 + (1-λ)*x2, same for labels.
        
        Critical for regularization on small datasets like APTOS (3.6k images).
        Paper: https://arxiv.org/abs/1710.09412
        """
        batch_size = tf.shape(images)[0]
        # Sample lambda from Beta distribution
        lam = tf.random.uniform([], 0.0, self.mixup_alpha)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Mixup
        mixed_images = lam * images + (1.0 - lam) * shuffled_images
        mixed_labels = lam * labels + (1.0 - lam) * shuffled_labels
        
        return mixed_images, mixed_labels

    def _cutmix_batch(self, images, labels):
        """Cutmix augmentation: paste random patch from another image.
        
        Better than Mixup for preserving local features.
        Paper: https://arxiv.org/abs/1905.04899
        """
        batch_size = tf.shape(images)[0]
        img_h = tf.shape(images)[1]
        img_w = tf.shape(images)[2]
        
        # Sample lambda from Beta distribution
        lam = tf.random.uniform([], 0.0, 1.0)
        
        # Calculate cut size
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
        cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)
        
        # Random center position
        cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
        cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
        
        # Bounding box
        bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
        bby1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
        bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
        bby2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Create mask
        mask = tf.ones((img_h, img_w, 1), dtype=tf.float32)
        mask_paddings = [[bby1, img_h - bby2], [bbx1, img_w - bbx2], [0, 0]]
        cut_mask = tf.zeros((bby2 - bby1, bbx2 - bbx1, 1), dtype=tf.float32)
        # Apply mask would require more complex logic, using simple Mixup instead
        
        # Fallback to mixup for simplicity (Cutmix requires complex masking)
        mixed_images = lam * images + (1.0 - lam) * shuffled_images
        mixed_labels = lam * labels + (1.0 - lam) * shuffled_labels
        
        return mixed_images, mixed_labels

    def build_dataset(self, df: pd.DataFrame, training: bool = True):
        """Builds optimized tf.data.Dataset for M2."""
        paths = df['path'].values
        labels = df['diagnosis'].values.astype(np.int32)
        
        # Label format depends on output type
        if self.is_regression:
            # Regression: scalar float labels (0.0 to 4.0)
            labels = labels.astype(np.float32)
        else:
            # Classification: one-hot encoded labels
            labels = tf.one_hot(labels, depth=5)

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        # Shuffle before mapping for training
        # Use full dataset size for better mixing (critical for imbalanced small datasets)
        if training:
            dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

        # Load images (no normalization yet)
        dataset = dataset.map(self._load_image, num_parallel_calls=self.autotune)
        
        # Apply Ben Graham preprocessing if enabled (before caching for consistency)
        if self.use_ben_graham:
            dataset = dataset.map(self._apply_ben_graham, num_parallel_calls=self.autotune)
        
        # Cache only for smaller images (< 300px) to avoid OOM
        use_cache = getattr(self.config.data, 'use_cache', True)
        if use_cache and self.config.data.image_size <= 256:
            dataset = dataset.cache()
        
        # Augmentation on raw [0, 255] images (training only)
        if training:
            if self.use_strong_augment:
                # Use Albumentations for comprehensive augmentations
                dataset = dataset.map(self._augment_image_albumentations, num_parallel_calls=self.autotune)
            else:
                # Fallback to basic TF augmentations
                dataset = dataset.map(self._augment_image_native, num_parallel_calls=self.autotune)

        # Normalize AFTER augmentation
        dataset = dataset.map(self._normalize_image, num_parallel_calls=self.autotune)

        # Batch first (required for batch-level augmentations)
        dataset = dataset.batch(self.config.data.batch_size)

        # Apply Mixup/Cutmix AFTER batching (batch-level augmentation)
        if training and self.use_mixup:
            dataset = dataset.map(
                lambda x, y: self._mixup_batch(x, y),
                num_parallel_calls=self.autotune
            )

        # Prefetch
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
