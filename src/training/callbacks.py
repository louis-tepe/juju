import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import wandb
from src.utils.metrics import quadratic_weighted_kappa, get_optimal_thresholds
from scipy.optimize import minimize

class QWKCallback(keras.callbacks.Callback):
    """
    Custom callback to compute Quadratic Weighted Kappa on Validation set
    at the end of each epoch.
    """
    def __init__(self, validation_data, batch_size=32):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        # Extract Images and Labels from validation dataset
        # This might be slow for large val sets, but necessary for global QWK
        
        # Note: tf.data.Dataset iteration
        val_images = []
        val_labels = []
        
        for images, labels in self.validation_data:
            val_labels.extend(labels.numpy())
            # We don't need to store images if we just predict
        
        val_labels = np.array(val_labels)
        if len(val_labels.shape) > 1 and val_labels.shape[-1] > 1:
             val_labels = np.argmax(val_labels, axis=-1)
        
        # Predict
        val_preds_raw = self.model.predict(self.validation_data, verbose=0)
        
        # 1. Standard QWK
        # If classification (softmax), take argmax
        if len(val_preds_raw.shape) > 1 and val_preds_raw.shape[-1] > 1:
             val_preds_rounded = np.argmax(val_preds_raw, axis=-1)
        else:
             # Regression
             val_preds_rounded = np.rint(val_preds_raw).astype(int)
             val_preds_rounded = np.clip(val_preds_rounded, 0, 4)
        
        qwk = quadratic_weighted_kappa(val_labels, val_preds_rounded)
        
        # 2. Optimized QWK (Threshold search)
        def _eval_qwk(thresholds):
            preds = pd.cut(val_preds_raw, [-np.inf] + list(thresholds) + [np.inf], labels=[0, 1, 2, 3, 4])
            return -quadratic_weighted_kappa(val_labels, preds)

        # Use scipy to find thresholds (simplified for speed here, usually done offline)
        # For logging, just logging standard QWK is often enough.
        
        print(f"\nEpoch {epoch+1}: val_qwk: {qwk:.4f}")
        
        # logs["val_qwk"] = qwk  <-- This is already there, just ensuring we rely on it
        logs["val_qwk"] = qwk

class LRLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "numpy"):
            lr = lr.numpy()
        elif hasattr(lr, "value"):
            lr = lr.value()
        logs["learning_rate"] = float(lr)
