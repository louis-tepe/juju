import tensorflow as tf


class KappaLoss(tf.keras.losses.Loss):
    """
    A differentiable approximation of the Quadratic Weighted Kappa.
    Designed for regression output (1 unit).
    """
    def __init__(self, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        # y_true: (batch, 1)
        # y_pred: (batch, 1) continuous

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to valid range
        y_pred = tf.clip_by_value(y_pred, 0, self.num_classes - 1)

        # Weight matrix W: (i-j)^2 / (N-1)^2
        # Here we calculate per-sample squared error directly which corresponds to numerator
        # The challenge is the denominator (Expected agreement).
        # Direct QWK optimization in batch is unstable.
        # Often people use simple MSE as a proxy for QWK numerator.

        # "Soft" Kappa Loss implementation varies.
        # A simple proxy is Mean Squared Error for Regression,
        # as QWK is related to 1 - MSE/Variance.
        # So minimizing MSE maximizes QWK for regression.

        # We will return MSE here as the robust baseline "proxy" for now,
        # but labeled as KappaLoss for architecture consistency if we want to swap later.
        # True "SoftKappa" requires Softmax output.

        return tf.reduce_mean(tf.square(y_true - y_pred))

def get_loss(name):
    if name == 'huber':
        return tf.keras.losses.Huber(delta=1.0)
    elif name == 'mse':
        return tf.keras.losses.MeanSquaredError()
    elif name == 'kappa':
        return KappaLoss()
    elif name == 'crossentropy':
        return tf.keras.losses.CategoricalCrossentropy()
    else:
        raise ValueError(f"Unknown loss: {name}")
