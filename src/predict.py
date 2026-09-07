import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from src.config import settings
from src.model_factory import build_model
from src.transforms import build_transforms
from src.utils import get_device


def load_production_model(model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(
        checkpoint["model_name"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_image(image_path: Path, model_path: Path):
    device = get_device()
    model, checkpoint = load_production_model(model_path, device)

    _, eval_transform = build_transforms(checkpoint["image_size"])
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    logit = model(tensor).squeeze()
    probability = torch.sigmoid(logit).item()
    threshold = float(checkpoint["threshold"])

    label = "malignant" if probability >= threshold else "benign"

    return {
        "label": label,
        "probability_malignant": probability,
        "threshold": threshold,
        "model_name": checkpoint["model_name"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=settings.production_dir / "model.pt",
    )
    args = parser.parse_args()

    result = predict_image(args.image, args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
