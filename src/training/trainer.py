import os

import hydra
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix

from src.data.loader import get_dataset
from src.models.factory import create_model
from src.models.losses import get_loss
from src.training.callbacks import LRLogger, QWKCallback


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = create_model(config)

    def train(self):
        # Data
        train_ds = get_dataset(self.config, split='train')
        val_ds = get_dataset(self.config, split='val')

        # Optimizer
        optimizer = keras.optimizers.get({
            "class_name": self.config.train.optimizer,
            "config": {"learning_rate": self.config.train.lr}
        })

        # Loss
        loss_fn = get_loss(self.config.train.loss)

        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy'] # Monitor Accuracy
        )

        # Callbacks
        callbacks = [
            QWKCallback(val_ds),
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


        # SWA: Start at 75% of epochs
        # swa_start = int(self.config.train.epochs * 0.75)
        # callbacks.append(SWA(start_epoch=swa_start))
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
        
        # Get backbone layer (layer index 1 after Input)
        backbone = self.model.layers[1]
        
        # ============ PHASE 1: Head Warmup (Frozen Backbone) ============
        print(f"\n{'='*50}")
        print(f"PHASE 1: Head Warmup ({warmup_epochs} epochs) - Backbone FROZEN")
        print(f"{'='*50}")
        
        backbone.trainable = False
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=self.config.train.lr),
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        # Callbacks without early stopping for warmup
        warmup_callbacks = [
            QWKCallback(val_ds),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(os.getcwd(), "model_best.keras"),
                monitor="val_qwk",
                mode="max",
                save_best_only=True,
                verbose=1
            ),
            LRLogger()
        ]
        
        history1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            callbacks=warmup_callbacks,
            class_weight=class_weights
        )
        
        # ============ PHASE 2: Full Fine-tuning (Unfrozen) ============
        finetune_epochs = total_epochs - warmup_epochs
        if finetune_epochs > 0:
            print(f"\n{'='*50}")
            print(f"PHASE 2: Fine-tuning ({finetune_epochs} epochs) - Backbone UNFROZEN")
            print(f"LR: {finetune_lr:.2e} (10x lower)")
            print(f"{'='*50}")
            
            backbone.trainable = True
            self.model.compile(
                optimizer=keras.optimizers.AdamW(learning_rate=finetune_lr),
                loss=loss_fn,
                metrics=['accuracy']
            )
            
            # Full callbacks with early stopping
            finetune_callbacks = [
                QWKCallback(val_ds),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(os.getcwd(), "model_best.keras"),
                    monitor="val_qwk",
                    mode="max",
                    save_best_only=True,
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
                ),
                LRLogger()
            ]
            
            history2 = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=finetune_epochs,
                callbacks=finetune_callbacks,
                class_weight=class_weights
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
        val_preds = []

        # Iterate to get labels and images
        # Note: map(lambda x, y: (x, y)) is implicit in dataset
        for images, labels in val_ds:
            preds = self.model.predict(images, verbose=0)
            val_labels.extend(np.argmax(labels.numpy(), axis=-1))
            val_preds.extend(np.argmax(preds, axis=-1))

        oof_df = pd.DataFrame({
            'y_true': val_labels,
            'y_pred': val_preds
        })
        oof_df.to_csv("oof.csv", index=False)
        print("Saved OOF predictions to oof.csv")

        # Plot Confusion Matrix
        self.plot_confusion_matrix(val_labels, val_preds)

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
