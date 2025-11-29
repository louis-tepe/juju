import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import wandb
import os
from src.data.loader import get_dataset
from src.models.factory import create_model
from src.models.losses import get_loss
from src.training.callbacks import QWKCallback
from src.training.swa import SWA

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
            metrics=['mse'] # Monitor MSE as proxy
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
        swa_start = int(self.config.train.epochs * 0.75)
        callbacks.append(SWA(start_epoch=swa_start))
        
        if wandb.run:
            callbacks.append(wandb.keras.WandbCallback(save_model=False))
            
        # Train
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.train.epochs,
            callbacks=callbacks
        )
        
        # Save OOF Predictions
        print("Generating OOF predictions...")
        val_labels = []
        val_preds = []
        
        # Iterate to get labels and images
        # Note: map(lambda x, y: (x, y)) is implicit in dataset
        for images, labels in val_ds:
            preds = self.model.predict(images, verbose=0)
            val_labels.extend(labels.numpy())
            val_preds.extend(preds.flatten())
            
        oof_df = pd.DataFrame({
            'y_true': val_labels,
            'y_pred': val_preds
        })
        oof_df.to_csv("oof.csv", index=False)
        print("Saved OOF predictions to oof.csv")
        
        return self.model
