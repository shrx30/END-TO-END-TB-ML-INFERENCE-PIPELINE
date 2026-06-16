
import torch
from timm import create_model


def load_model():
    device = torch.device("cpu")  # keep CPU for now

    model = create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=2
    )

    model.load_state_dict(
        torch.load(
            "models/vit_model.pth",
            map_location=device
        )
    )

    model.eval()
    return model

