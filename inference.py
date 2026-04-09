# app/inference.py

from PIL import Image
import torch
import io
import torchvision.transforms as transforms

# ViT preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


import torch.nn.functional as F

def predict(model, image_bytes):
    input_tensor = preprocess(image_bytes)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)   # convert to probabilities
        predicted_class = torch.argmax(probs, dim=1)

    return {
        "class": int(predicted_class.item()),
        "confidence": float(torch.max(probs).item())
    }