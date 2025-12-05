import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
class GeMPooling2D(keras.layers.Layer):
    """
    Generalized Mean Pooling layer.
    """
    def __init__(self, p=3.0, train_p=True, **kwargs):
        super().__init__(**kwargs)
        self.p_init = p
        self.train_p = train_p

    def build(self, input_shape):
        self.p = self.add_weight(
            name="p",
            shape=(1,),
            initializer=keras.initializers.Constant(self.p_init),
            trainable=self.train_p,
            dtype=tf.float32
        )
        super().build(input_shape)


    def call(self, inputs):
        # Ensure p is within reasonable bounds to prevent overflow/underflow
        p = tf.clip_by_value(self.p, 1.0, 10.0)

        # GeM is typically defined for positive activations.
        # EfficientNet uses Swish which can be negative. We use ReLU to enforce positivity.
        x = tf.nn.relu(inputs)

        # Add epsilon for numerical stability
        x = tf.maximum(x, 1e-6)

        x = tf.pow(x, p)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        x = tf.pow(x, 1.0 / p)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p_init,
            "train_p": self.train_p
        })
        return config


@keras.saving.register_keras_serializable()
class ScaleLayer(keras.layers.Layer):
    """
    Simple scaling layer that multiplies input by a constant factor.
    Used for constrained regression output: sigmoid * scale → output in [0, scale].
    
    This is a serializable replacement for Lambda(lambda x: x * scale).
    """
    def __init__(self, scale=4.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config
