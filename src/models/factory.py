import keras

from src.models.layers import GeMPooling2D, ScaleLayer


def get_backbone(name, input_shape, pretrained=True):
    """
    Factory to get the backbone model (excluding top).
    """
    weights = 'imagenet' if pretrained else None

    if 'efficientnet' in name:
        # Dynamically map efficientnet names
        if name == 'efficientnet_b0':
            return keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
        elif name == 'efficientnet_b1':
            return keras.applications.EfficientNetB1(include_top=False, weights=weights, input_shape=input_shape)
        elif name == 'efficientnet_b2':
            return keras.applications.EfficientNetB2(include_top=False, weights=weights, input_shape=input_shape)
        elif name == 'efficientnet_b3':
            return keras.applications.EfficientNetB3(include_top=False, weights=weights, input_shape=input_shape)
        elif name == 'efficientnet_b4':
            return keras.applications.EfficientNetB4(include_top=False, weights=weights, input_shape=input_shape)
        elif name == 'efficientnet_b5':
            return keras.applications.EfficientNetB5(include_top=False, weights=weights, input_shape=input_shape)
    elif 'resnet50' in name:
        return keras.applications.ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    elif 'inception_resnet_v2' in name:
        return keras.applications.InceptionResNetV2(include_top=False, weights=weights, input_shape=input_shape)

    raise ValueError(f"Unknown backbone: {name}")

def create_model(config):
    """
    Constructs the full model from config.
    """
    input_shape = (config.data.image_size, config.data.image_size, 3)

    backbone = get_backbone(config.model.backbone, input_shape, config.model.pretrained)
    
    # Freeze backbone if specified (prevents destroying pretrained weights)
    freeze_backbone = getattr(config.model, 'freeze_backbone', False)
    if freeze_backbone:
        backbone.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = backbone(inputs)

    # Pooling
    if config.model.head == 'gem':
        x = GeMPooling2D()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)

    # Dropout
    if config.model.dropout > 0:
        x = keras.layers.Dropout(config.model.dropout)(x)

    # Head: support multiple output types
    output_type = getattr(config.model, 'output_type', 'softmax')
    constrain_regression = getattr(config.model, 'constrain_regression', True)
    
    if config.model.use_ordinal:
        # Legacy: Ordinal Regression (5 sigmoid units)
        outputs = keras.layers.Dense(config.model.num_classes, activation='sigmoid', name='output')(x)
    elif output_type == 'regression':
        if constrain_regression:
            # Constrained regression: sigmoid * 4 → output in [0, 4]
            # This helps the model converge faster and prevents extreme predictions
            # Using ScaleLayer instead of Lambda for serialization compatibility
            x = keras.layers.Dense(1, activation='sigmoid', name='pre_output')(x)
            outputs = ScaleLayer(scale=4.0, name='output')(x)
        else:
            # Unconstrained regression: linear output
            outputs = keras.layers.Dense(1, activation='linear', name='output')(x)
    else:
        # Classification: Softmax (default)
        outputs = keras.layers.Dense(config.model.num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs)
    return model
