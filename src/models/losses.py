import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
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


@keras.saving.register_keras_serializable(name="CategoricalFocalCrossentropy")
class CategoricalFocalCrossentropy(tf.keras.losses.Loss):
    """
    Categorical Focal Crossentropy Loss.
    References:
        - [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
    """
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow((1 - y_pred), self.gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing
        })
        return config


def get_loss(name, **kwargs):
    """
    Get loss function by name.
    """
    delta = kwargs.get('delta', 0.5)
    label_smoothing = kwargs.get('label_smoothing', 0.0)
    
    if name == 'huber' or name == 'smooth_l1':
        return tf.keras.losses.Huber(delta=delta)
    elif name == 'mse':
        return tf.keras.losses.MeanSquaredError()
    elif name == 'mae':
        return tf.keras.losses.MeanAbsoluteError()
    elif name == 'kappa':
        return KappaLoss()
    elif name == 'crossentropy':
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    elif name == 'focal':
        return CategoricalFocalCrossentropy(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0),
            label_smoothing=label_smoothing
        )
    else:
        raise ValueError(f"Unknown loss: {name}")
