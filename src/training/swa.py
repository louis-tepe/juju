import keras


class SWA(keras.callbacks.Callback):
    """
    Stochastic Weight Averaging (SWA) Callback.
    Averages model weights across epochs to improve generalization.
    """
    def __init__(self, start_epoch, swa_freq=1):
        super().__init__()
        self.start_epoch = start_epoch
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.num_models = 0

    def on_train_begin(self, logs=None):
        self.swa_weights = None
        self.num_models = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.swa_freq == 0:
            current_weights = self.model.get_weights()

            if self.swa_weights is None:
                self.swa_weights = current_weights
            else:
                # Running average: (n * avg + current) / (n + 1)
                # But standard SWA is usually simple average at end,
                # or moving average.
                # Standard formula: w_swa = (w_swa * n + w) / (n + 1)

                self.swa_weights = [
                    (swa_w * self.num_models + cur_w) / (self.num_models + 1)
                    for swa_w, cur_w in zip(self.swa_weights, current_weights, strict=False)
                ]

            self.num_models += 1
            print(f"\n[SWA] Updated SWA weights (n={self.num_models})")

    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            print("\n[SWA] Setting final model weights to SWA average.")
            self.model.set_weights(self.swa_weights)

            # Recompute BatchNorm statistics (Optional but recommended for SWA)
            # Note: Keras doesn't have an easy built-in for this without iterating data again.
            # For simplicity in this implementation, we skip BN update or assume
            # the last batch metrics are close enough, or user runs a separate pass.
            # Proper SWA requires a forward pass on train data to update BN running mean/var.
