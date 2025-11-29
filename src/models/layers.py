import tensorflow as tf
import keras

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
        # Avoid 0 or negative values for power operation stability if needed, 
        # but usually inputs are ReLU activated features >= 0.
        # To be safe against 0, we can add epsilon.
        x = tf.maximum(inputs, 1e-6)
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        x = tf.pow(x, 1.0 / self.p)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p_init,
            "train_p": self.train_p
        })
        return config
