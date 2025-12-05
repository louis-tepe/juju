import os
from functools import partial

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.preprocess import load_and_preprocess_image


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.autotune = tf.data.AUTOTUNE

    def get_augmentations(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
        ])

    def _process_path(self, path, label, size, use_ben_graham):
        """Load and preprocess a single image."""
        def _read_img(path_b, size_b, use_bg_b):
            path_str = path_b.decode("utf-8")
            size_int = int(size_b)
            use_bg_bool = bool(use_bg_b)

            try:
                if use_bg_bool:
                    img = load_and_preprocess_image(path_str, size=size_int)
                else:
                    import cv2
                    img = cv2.imread(path_str)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (size_int, size_int))
                    img = img.astype(np.float32) / 255.0
                
                img = img.astype(np.float32)
                return img
            except Exception as e:
                print(f"Error processing {path_str}: {e}")
                return np.zeros((size_int, size_int, 3), dtype=np.float32)

        img = tf.numpy_function(
            _read_img,
            [path, size, use_ben_graham],
            tf.float32
        )
        img.set_shape([size, size, 3])
        return img, label

    def _augment_image(self, image, label):
        """Albumentations-based augmentation (slower, more flexible)."""
        def _aug_fn(img):
            aug = self.get_augmentations()
            data = aug(image=img)
            return data['image']

        aug_img = tf.numpy_function(func=_aug_fn, inp=[image], Tout=tf.float32)
        aug_img.set_shape([self.config.data.image_size, self.config.data.image_size, 3])
        return aug_img, label

    def _augment_image_native(self, image, label):
        """TF-Native augmentations (faster, GPU-friendly)."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    def build_dataset(self, df: pd.DataFrame, training: bool = True):
        """Builds a tf.data.Dataset from a pandas DataFrame."""
        paths = df['path'].values
        labels = df['diagnosis'].values.astype(np.int32)
        labels = tf.one_hot(labels, depth=5)

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        process_fn = partial(
            self._process_path,
            size=self.config.data.image_size,
            use_ben_graham=self.config.data.use_ben_graham
        )

        dataset = dataset.map(process_fn, num_parallel_calls=4)
        
        if training:
            use_native = hasattr(self.config.data, 'use_native_augment') and self.config.data.use_native_augment
            if use_native:
                dataset = dataset.map(self._augment_image_native, num_parallel_calls=4)
            else:
                dataset = dataset.map(self._augment_image, num_parallel_calls=4)

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
