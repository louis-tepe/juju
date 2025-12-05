import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from src.models.factory import create_model


@pytest.fixture
def config():
    return OmegaConf.create({
        "data": {"image_size": 256},
        "model": {
            "backbone": "efficientnet_b0",
            "pretrained": False, # Faster for test
            "num_classes": 5,
            "dropout": 0.1,
            "head": "gem",
            "use_ordinal": False
        }
    })

def test_model_creation(config):
    model = create_model(config)
    assert isinstance(model, tf.keras.Model)

    input_shape = (1, 256, 256, 3)
    output = model(tf.random.normal(input_shape))

    # Classification output: 5 classes (softmax)
    assert output.shape == (1, 5)

def test_gem_pooling():
    from src.models.layers import GeMPooling2D
    layer = GeMPooling2D(p=3.0)
    x = tf.random.normal((1, 8, 8, 1280)) # Typical EfficientNet feature map
    y = layer(x)
    assert y.shape == (1, 1280)
