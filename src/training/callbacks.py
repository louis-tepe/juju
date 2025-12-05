import keras
import numpy as np

from src.utils.metrics import quadratic_weighted_kappa


class QWKCallback(keras.callbacks.Callback):
    """
    Custom callback to compute Quadratic Weighted Kappa on Validation set
    at the end of each epoch.
    
    Supports both regression (single neuron) and classification (softmax) outputs.
    Pre-extracts validation data to avoid dataset exhaustion issues.
    """
    def __init__(self, validation_data, is_regression=False):
        super().__init__()
        self.is_regression = is_regression
        
        # Pre-extract all validation data once to avoid dataset exhaustion
        print("QWKCallback: Pre-extracting validation data...")
        self.val_images = []
        self.val_labels = []
        for images, labels in validation_data:
            self.val_images.append(images.numpy())
            self.val_labels.append(labels.numpy())
        
        self.val_images = np.concatenate(self.val_images, axis=0)
        self.val_labels = np.concatenate(self.val_labels, axis=0)
        
        # Convert labels to class indices
        if is_regression:
            # Regression mode: labels are already scalar (0.0-4.0), just convert to int
            self.val_labels = np.rint(self.val_labels).astype(np.int64)
        elif len(self.val_labels.shape) > 1 and self.val_labels.shape[-1] > 1:
            # Classification mode: one-hot encoded, convert to class indices
            self.val_labels = np.argmax(self.val_labels, axis=-1).astype(np.int64)
        else:
            # Fallback: ensure int type
            self.val_labels = self.val_labels.astype(np.int64)
        
        print(f"QWKCallback: Loaded {len(self.val_labels)} validation samples")
        print(f"QWKCallback: Mode = {'REGRESSION' if is_regression else 'CLASSIFICATION'}")
        print(f"QWKCallback: Label distribution = {np.bincount(self.val_labels, minlength=5)}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Predict in batches to avoid memory issues
        # Use very small batch size for B5 + Float32 to prevent OOM
        batch_size = 2
        val_preds = []
        for i in range(0, len(self.val_images), batch_size):
            batch = self.val_images[i:i+batch_size]
            preds = self.model(batch, training=False)
            val_preds.append(preds.numpy())
        
        val_preds_raw = np.concatenate(val_preds, axis=0)

        # Convert predictions to class indices based on output type
        if self.is_regression:
            # Regression: round and clip to valid class range
            val_preds_rounded = np.rint(val_preds_raw.flatten()).astype(int)
            val_preds_rounded = np.clip(val_preds_rounded, 0, 4)
        elif len(val_preds_raw.shape) > 1 and val_preds_raw.shape[-1] > 1:
            # Classification: argmax
            val_preds_rounded = np.argmax(val_preds_raw, axis=-1)
        else:
            # Fallback: round
            val_preds_rounded = np.rint(val_preds_raw).astype(int)
            val_preds_rounded = np.clip(val_preds_rounded, 0, 4)

        qwk = quadratic_weighted_kappa(self.val_labels, val_preds_rounded)

        # Diagnostic: show prediction distribution to detect issues
        pred_dist = np.bincount(val_preds_rounded, minlength=5)
        
        if self.is_regression:
            pred_mean = np.mean(val_preds_raw)
            pred_std = np.std(val_preds_raw)
            print(f"\nEpoch {epoch+1}: val_qwk: {qwk:.4f} | pred_mean: {pred_mean:.2f} | pred_std: {pred_std:.2f}")
        else:
            # For classification, show confidence stats
            max_probs = np.max(val_preds_raw, axis=-1)
            print(f"\nEpoch {epoch+1}: val_qwk: {qwk:.4f} | mean_conf: {np.mean(max_probs):.2f} | min_conf: {np.min(max_probs):.2f}")
        
        print(f"Prediction distribution: {pred_dist} | True: {np.bincount(self.val_labels, minlength=5)}")
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
