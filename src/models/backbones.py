SUPPORTED_BACKBONES = ("MobileNetV2", "EfficientNetB0", "ResNet50")


def get_backbone(name, input_shape, weights="imagenet"):
    """Return a Keras application model and its matching preprocessing function."""
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone {name!r}. Choose from: "
            + ", ".join(SUPPORTED_BACKBONES)
        )

    if name == "MobileNetV2":
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        constructor = MobileNetV2
    elif name == "EfficientNetB0":
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input
        constructor = EfficientNetB0
    else:
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        constructor = ResNet50

    model = constructor(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    return model, preprocess_input


def get_preprocess_input(name):
    """Load preprocessing without constructing the backbone."""
    if name == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    elif name == "EfficientNetB0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        raise ValueError(f"Unsupported backbone {name!r}.")
    return preprocess_input
