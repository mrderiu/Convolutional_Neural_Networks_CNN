import torch

from src.model_factory import build_model


def test_resnet18_binary_output():
    model = build_model("resnet18", pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1)
