import os
import json

import hydra
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from src.data.loader import get_dataset
from src.models.factory import create_model
from src.models.losses import get_loss
from src.training.callbacks import LRLogger, QWKCallback
from src.training.swa import SWA


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = create_model(config)

    def train(self):
        # Detect output type for loss selection
        output_type = getattr(self.config.model, 'output_type', 'softmax')
        is_regression = output_type == 'regression'
        
        # Data
        train_ds = get_dataset(self.config, split='train')
        val_ds = get_dataset(self.config, split='val')

        # Loss: Use Huber (Smooth L1) for regression, CrossEntropy for classification
        # Huber loss is more robust to mislabeled outliers (1st place APTOS solution insight)
        if is_regression:
            huber_delta = getattr(self.config.train, 'huber_delta', 0.5)
            loss_fn = keras.losses.Huber(delta=huber_delta)
            print(f"📊 Using REGRESSION mode (Huber loss, delta={huber_delta}) for threshold optimization")
        else:
            loss_fn = get_loss(self.config.train.loss)
            print("📊 Using CLASSIFICATION mode (CrossEntropy loss)")

        # Metrics depend on output type
        metrics = ['mae'] if is_regression else ['accuracy']


        # Callbacks
        callbacks = [
            QWKCallback(val_ds, is_regression=is_regression),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(os.getcwd(), "model_best.keras"),
                monitor="val_qwk",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=self.config.train.min_lr,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_qwk",
                mode="max",
                patience=self.config.train.patience,
                restore_best_weights=True,
                verbose=1
            )
        ]


        callbacks.append(LRLogger())

        # WandB Integration (Decoupled)
        if wandb.run:
            try:
                from wandb.integration.keras import WandbMetricsLogger
                callbacks.append(WandbMetricsLogger())
            except ImportError:
                print("WandB not installed or not active. Skipping WandB callback.")

        # Calculate Class Weights
        from sklearn.utils import class_weight

        train_df_path = hydra.utils.to_absolute_path(self.config.data.train_folds_csv)
        if not os.path.exists(train_df_path):
             train_df_path = hydra.utils.to_absolute_path(self.config.data.train_csv)

        df = pd.read_csv(train_df_path)
        
        if 'fold' in df.columns:
            df_train = df[df['fold'] != self.config.train.fold]
        else:
            df_train = df
            
        y_train = df_train['diagnosis'].values

        class_weights_vals = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights_vals))
        print(f"Class Weights: {class_weights}")

        # Two-phase training configuration
        warmup_epochs = getattr(self.config.train, 'warmup_epochs', 3)
        total_epochs = self.config.train.epochs
        finetune_lr = getattr(self.config.train, 'finetune_lr', self.config.train.lr * 0.1)
        use_cosine_decay = getattr(self.config.train, 'use_cosine_decay', False)
        
        # Get backbone layer (layer index 1 after Input)
        backbone = self.model.layers[1]
        
        # ============ PHASE 1: Head Warmup (Frozen Backbone) ============
        print(f"\n{'='*50}")
        print(f"PHASE 1: Head Warmup ({warmup_epochs} epochs) - Backbone FROZEN")
        print(f"{'='*50}")
        
        backbone.trainable = False
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=self.config.train.lr, clipnorm=1.0),
            loss=loss_fn,
            metrics=metrics
        )
        
        # Callbacks without early stopping for warmup
        # Save both full model (.keras) and weights only (.h5) for faster Phase 2 loading
        warmup_callbacks = [
            QWKCallback(val_ds, is_regression=is_regression),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(os.getcwd(), "model_best.keras"),
                monitor="val_qwk",
                mode="max",
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(os.getcwd(), "model_best.weights.h5"),
                monitor="val_qwk",
                mode="max",
                save_best_only=True,
                save_weights_only=True,  # Faster to load in Phase 2
                verbose=0
            ),
            LRLogger()
        ]
        
        history1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            callbacks=warmup_callbacks,
            class_weight=class_weights if not is_regression else None
        )
        
        # ============ PHASE 2: Full Fine-tuning (Unfrozen) ============
        finetune_epochs = total_epochs - warmup_epochs
        if finetune_epochs > 0:
            print(f"\n{'='*50}")
            print(f"PHASE 2: Fine-tuning ({finetune_epochs} epochs) - Backbone UNFROZEN")
            print(f"LR: {finetune_lr:.2e} (10x lower)")
            if use_cosine_decay:
                print("Using Cosine Decay scheduler")
            print(f"{'='*50}")
            
            # Load best weights from Phase 1 (weights only - faster than full model)
            best_weights_path = os.path.join(os.getcwd(), "model_best.weights.h5")
            best_model_path = os.path.join(os.getcwd(), "model_best.keras")
            
            if os.path.exists(best_weights_path):
                print(f"Loading best weights from Phase 1: {best_weights_path}")
                self.model.load_weights(best_weights_path)
            elif os.path.exists(best_model_path):
                # Fallback: load from full model (slower)
                print(f"Loading best model from Phase 1: {best_model_path}")
                loaded_model = keras.models.load_model(best_model_path)
                self.model.set_weights(loaded_model.get_weights())
                del loaded_model  # Free memory
            
            backbone.trainable = True
            
            # OPTIMIZATION: Keep BatchNormalization layers frozen!
            # 1. Preserves pretrained statistics (critical for small batch sizes)
            # 2. Significantly speeds up training on Metal/M2 (avoids updating BN params)
            for layer in backbone.layers:
                if isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = False
            
            # Optimizer: Cosine decay or constant LR
            if use_cosine_decay:
                # Estimate steps per epoch
                steps_per_epoch = len(train_ds)
                decay_steps = steps_per_epoch * finetune_epochs
                lr_schedule = keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=finetune_lr,
                    decay_steps=decay_steps,
                    alpha=self.config.train.min_lr / finetune_lr
                )
                # Remove clipnorm=1.0 - global norm calculation is expensive on Metal with full backbone
                optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule)
            else:
                optimizer = keras.optimizers.AdamW(learning_rate=finetune_lr)
            
            # Loss for fine-tuning: keep same as phase 1 for regression
            if is_regression:
                finetune_loss = loss_fn
            else:
                label_smoothing = getattr(self.config.train, 'label_smoothing', 0.1)
                finetune_loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
            
            # Recompile with new optimizer and frozen BN
            self.model.compile(
                optimizer=optimizer,
                loss=finetune_loss,
                metrics=metrics
            )
            
            # Full callbacks with early stopping
            finetune_callbacks = [
                QWKCallback(val_ds, is_regression=is_regression),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(os.getcwd(), "model_best.keras"),
                    monitor="val_qwk",
                    mode="max",
                    save_best_only=True,
                    verbose=1
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_qwk",
                    mode="max",
                    patience=self.config.train.patience,
                    restore_best_weights=False,  # SWA handles final weights
                    verbose=1
                ),
                LRLogger()
            ]
            
            # Only add ReduceLROnPlateau if not using cosine decay
            if not use_cosine_decay:
                finetune_callbacks.insert(2, keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=2,
                    min_lr=self.config.train.min_lr,
                    verbose=1
                ))
            
            # Add SWA callback if enabled
            use_swa = getattr(self.config.train, 'use_swa', False)
            if use_swa:
                swa_start = getattr(self.config.train, 'swa_start_epoch', int(total_epochs * 0.75))
                # Adjust for phase 2 (swa_start is relative to total epochs)
                swa_start_phase2 = max(0, swa_start - warmup_epochs)
                finetune_callbacks.append(SWA(start_epoch=swa_start_phase2))
                print(f"SWA enabled: will start averaging at epoch {swa_start} (phase 2 epoch {swa_start_phase2})")
            
            history2 = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=finetune_epochs,
                callbacks=finetune_callbacks,
                class_weight=class_weights if not is_regression else None
            )
            
            # Merge histories for plotting
            for key in history1.history:
                history1.history[key].extend(history2.history.get(key, []))
        
        history = history1

        # Plot Metrics
        self.plot_metrics(history)

        # Save OOF Predictions
        print("Generating OOF predictions...")
        val_labels = []
        val_preds_raw = []

        # Iterate to get labels and images
        for images, labels in val_ds:
            preds = self.model.predict(images, verbose=0)
            
            # Labels: scalar for regression, one-hot for classification
            if is_regression:
                val_labels.extend(labels.numpy().astype(int))
                val_preds_raw.extend(preds.flatten())
            else:
                val_labels.extend(np.argmax(labels.numpy(), axis=-1))
                val_preds_raw.extend(np.argmax(preds, axis=-1))

        val_labels = np.array(val_labels)
        val_preds_raw = np.array(val_preds_raw)
        
        # Save raw predictions for threshold optimization
        oof_df = pd.DataFrame({
            'y_true': val_labels,
            'y_pred': val_preds_raw
        })
        oof_df.to_csv("oof.csv", index=False)
        print("Saved OOF predictions to oof.csv")

        # Compute baseline QWK (simple rounding for regression, direct for classification)
        if is_regression:
            val_preds_baseline = np.clip(np.rint(val_preds_raw), 0, 4).astype(int)
        else:
            val_preds_baseline = val_preds_raw.astype(int)
        
        baseline_qwk = cohen_kappa_score(val_labels, val_preds_baseline, weights='quadratic')
        print(f"\n📊 Baseline QWK (simple rounding): {baseline_qwk:.4f}")
        
        # Threshold optimization (for regression mode)
        if is_regression:
            print("🔧 Running threshold optimization (multi-start)...")
            
            def kappa_loss(thresholds, X, y):
                """Negative QWK for minimization."""
                thresholds = np.sort(thresholds)
                preds = np.digitize(X, thresholds)
                return -cohen_kappa_score(y, preds, weights='quadratic')
            
            # Initial thresholds from 1st place APTOS solution: [0.7, 1.5, 2.5, 3.5]
            # The key insight was that the boundary between class 0 and 1 should be higher
            # Initial thresholds from top APTOS solutions
            # Lower first threshold (0.5 instead of 0.7) helps with class 0/1 boundary
            initial_thresholds = [0.5, 1.5, 2.5, 3.5]
            
            # Multi-start optimization to avoid local minima
            # Increased from 10 to 50 restarts for better convergence
            best_thresholds = np.array(initial_thresholds)
            best_kappa = -1
            n_restarts = 50
            
            for i in range(n_restarts):
                if i == 0:
                    # First try: use initial thresholds from top solution
                    start = np.array(initial_thresholds)
                else:
                    # Random restarts with noise around initial thresholds
                    noise = np.random.uniform(-0.3, 0.3, 4)
                    start = np.sort(np.clip(np.array(initial_thresholds) + noise, 0.1, 3.9))
                
                result = minimize(
                    kappa_loss,
                    start,
                    args=(val_preds_raw, val_labels),
                    method='nelder-mead',
                    options={'maxiter': 500, 'xatol': 1e-4}
                )
                
                candidate_thresholds = np.sort(result.x)
                candidate_preds = np.digitize(val_preds_raw, candidate_thresholds)
                candidate_kappa = cohen_kappa_score(val_labels, candidate_preds, weights='quadratic')
                
                if candidate_kappa > best_kappa:
                    best_kappa = candidate_kappa
                    best_thresholds = candidate_thresholds
            
            optimized_thresholds = best_thresholds
            
            # Apply optimized thresholds
            val_preds_opt = np.digitize(val_preds_raw, optimized_thresholds)
            optimized_qwk = cohen_kappa_score(val_labels, val_preds_opt, weights='quadratic')
            
            print(f"✅ Optimized thresholds: {optimized_thresholds.round(3).tolist()}")
            print(f"✅ Optimized QWK: {optimized_qwk:.4f} (improvement: +{optimized_qwk - baseline_qwk:.4f})")
            
            # Save thresholds
            thresholds_dict = {"thresholds": optimized_thresholds.tolist()}
            with open("thresholds.json", 'w') as f:
                json.dump(thresholds_dict, f)
            print("Saved optimized thresholds to thresholds.json")
            
            # Use optimized predictions for confusion matrix
            val_preds_final = val_preds_opt
        else:
            val_preds_final = val_preds_baseline

        # Plot Confusion Matrix
        self.plot_confusion_matrix(val_labels, val_preds_final)

        return self.model

    def plot_metrics(self, history):
        """Plots Loss, Accuracy and QWK per epoch."""
        metrics = history.history
        epochs = range(1, len(metrics['loss']) + 1)

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot Loss
        axes[0].plot(epochs, metrics['loss'], label='Train Loss')
        if 'val_loss' in metrics:
            axes[0].plot(epochs, metrics['val_loss'], label='Val Loss')
        axes[0].set_title('Loss per Epoch')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # Plot Accuracy
        if 'accuracy' in metrics:
            axes[1].plot(epochs, metrics['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in metrics:
                axes[1].plot(epochs, metrics['val_accuracy'], label='Val Accuracy')
            axes[1].set_title('Accuracy per Epoch')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()

        # Plot QWK
        if 'val_qwk' in metrics:
            axes[2].plot(epochs, metrics['val_qwk'], label='Val QWK', color='green')
            axes[2].set_title('QWK per Epoch')
            axes[2].set_xlabel('Epochs')
            axes[2].set_ylabel('QWK')
            axes[2].legend()

        plt.tight_layout()
        plt.savefig("metrics_plot.png")
        plt.close()
        print("Saved metrics plot to metrics_plot.png")

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plots Confusion Matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("confusion_matrix.png")
        plt.close()
        print("Saved confusion matrix to confusion_matrix.png")
