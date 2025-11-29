import tensorflow as tf
import numpy as np
import pandas as pd
import os
import albumentations as A
from functools import partial
from src.data.preprocess import preprocess_image

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
        """
        Wrapper for numpy processing to be used with tf.py_function
        """
        def _read_img(path_b, size_b, use_bg_b):
            path_str = path_b.decode("utf-8")
            size_int = int(size_b)
            use_bg_bool = bool(use_bg_b)
            
            try:
                if use_bg_bool:
                    img = preprocess_image(path_str, size=size_int)
                else:
                    # Standard resize
                    import cv2
                    img = cv2.imread(path_str)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (size_int, size_int))
                
                img = img.astype(np.float32) / 255.0
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
        """
        Wrapper for Albumentations
        """
        def _aug_fn(img):
            aug = self.get_augmentations()
            data = aug(image=img)
            return data['image']

        aug_img = tf.numpy_function(func=_aug_fn, inp=[image], Tout=tf.float32)
        aug_img.set_shape([self.config.data.image_size, self.config.data.image_size, 3])
        return aug_img, label

    def build_dataset(self, df: pd.DataFrame, training: bool = True):
        """
        Builds a tf.data.Dataset from a pandas DataFrame.
        Expected columns: 'path', 'diagnosis'
        """
        paths = df['path'].values
        labels = df['diagnosis'].values.astype(np.int32)
        
        # Create Dataset
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        
        # Map Loading & Preprocessing
        process_fn = partial(
            self._process_path, 
            size=self.config.data.image_size, 
            use_ben_graham=self.config.data.use_ben_graham
        )
        
        dataset = dataset.map(process_fn, num_parallel_calls=self.config.data.num_workers)
        
        if training:
            dataset = dataset.map(self._augment_image, num_parallel_calls=self.config.data.num_workers)
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(self.config.data.batch_size)
        
        if training:
            dataset = dataset.map(self.mixup, num_parallel_calls=self.autotune)
            
        dataset = dataset.prefetch(self.autotune)
        
        return dataset

    def mixup(self, images, labels, alpha=0.2):
        """
        Applies Mixup augmentation: x = lambda * x1 + (1 - lambda) * x2
        """
        batch_size = tf.shape(images)[0]
        
        # Sample lambda from Beta distribution
        # TF doesn't have Beta, so we use Gamma: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
        beta_dist = tf.random.gamma(shape=[batch_size], alpha=alpha)
        beta_dist2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
        lam = beta_dist / (beta_dist + beta_dist2)
        lam = tf.reshape(lam, [-1, 1, 1, 1])
        
        # Shuffle
        indices = tf.random.shuffle(tf.range(batch_size))
        images_shuffled = tf.gather(images, indices)
        labels_shuffled = tf.gather(labels, indices)
        
        # Mix Images
        images_mix = lam * images + (1 - lam) * images_shuffled
        
        # Mix Labels (Regression)
        labels = tf.cast(labels, tf.float32)
        labels_shuffled = tf.cast(labels_shuffled, tf.float32)
        
        if len(labels.shape) == 1:
             labels = tf.expand_dims(labels, -1)
             labels_shuffled = tf.expand_dims(labels_shuffled, -1)
        
        lam_label = tf.reshape(lam, [-1, 1])
        labels_mix = lam_label * labels + (1 - lam_label) * labels_shuffled
        
        return images_mix, labels_mix

def get_dataset(config, split='train'):
    """
    Factory function to get dataset based on split.
    """
    base_dir = config.data.train_images if split in ['train', 'val'] else config.data.test_images
    
    if split in ['train', 'val']:
        csv_path = config.data.train_folds_csv
        # Fallback if folds not created
        if not os.path.exists(csv_path):
            print(f"Folds file {csv_path} not found, using raw {config.data.train_csv}")
            csv_path = config.data.train_csv
            df = pd.read_csv(csv_path)
            # Simple random split fallback
            if 'fold' not in df.columns:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
                df['fold'] = -1
                for fold, (t, v) in enumerate(skf.split(df, df['diagnosis'])):
                    df.loc[v, 'fold'] = fold
        else:
            df = pd.read_csv(csv_path)
            
        # Filter by fold
        fold = config.train.fold
        if split == 'train':
            df = df[df['fold'] != fold]
        elif split == 'val':
            df = df[df['fold'] == fold]
    else:
        # Test
        csv_path = config.data.test_csv
        if not os.path.exists(csv_path):
             # Dummy df for running without data
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
