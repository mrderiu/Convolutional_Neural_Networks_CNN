import torch.nn as nn
from torchvision import models


AVAILABLE_MODELS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet152",
    "efficientnet_b0",
    "densenet121",
)


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Returns a binary classifier with ONE output logit.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = _binary_head(in_features)

    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = _binary_head(in_features)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = _binary_head(in_features)

    elif model_name == "resnet152":
        weights = models.ResNet152_Weights.DEFAULT if pretrained else None
        model = models.resnet152(weights=weights)
        in_features = model.fc.in_features
        model.fc = _binary_head(in_features)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {AVAILABLE_MODELS}"
        )

    return model


def _binary_head(in_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.40),
        nn.Linear(256, 1),
    )


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """
    Freezes feature extractor and keeps only the classification head trainable.
    """
    for param in model.parameters():
        param.requires_grad = False

    model_name = model_name.lower()

    if model_name.startswith("resnet"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "densenet121":
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(model_name)


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
