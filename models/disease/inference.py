"""
Disease model inference — ResNet-50 transfer learning.

Class names always loaded from checkpoint — never hardcoded (TR-15, TR-16).
val_transform used for inference — never train_transform.
Output schema: MASTER_SPEC §3.2 (frozen)
"""

import base64
import time
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# val_transform — NEVER use train_transform in inference (TR-16)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _build_model(num_classes: int) -> nn.Module:
    """Rebuild ResNet-50 architecture (must match train.py TR-13)."""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


class DiseaseInference:
    """Loads ResNet-50 checkpoint and runs disease classification."""

    def __init__(self, model_path: str) -> None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Disease model not found: {p}. Run train.py first."
            )

        # class_names always from checkpoint — never hardcoded (TR-15)
        checkpoint = torch.load(str(p), map_location="cpu")
        self.class_names: list[str] = checkpoint["class_names"]

        self._model = _build_model(len(self.class_names))
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        self._transform = val_transform  # never train_transform

        self.model_name = "resnet50_disease"
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def predict(self, input_data: dict) -> dict:
        """Input/output per MASTER_SPEC §3.2."""
        # Decode base64 image
        image_bytes = base64.b64decode(input_data["image_b64"])
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self._transform(image).unsqueeze(0)

        t_start = time.perf_counter()
        with torch.no_grad():
            logits = self._model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().tolist()
        inference_ms = (time.perf_counter() - t_start) * 1000.0

        class_probs = {cls: round(float(p), 4)
                       for cls, p in zip(self.class_names, probs)}
        top_prediction  = max(class_probs, key=class_probs.get)
        top_confidence  = class_probs[top_prediction]

        # Warn if probabilities don't sum to ≈1 (don't crash)
        prob_sum = sum(class_probs.values())
        if not (0.999 <= prob_sum <= 1.001):
            import logging
            logging.getLogger("disease-inference").warning(
                "class_probabilities sum=%.4f (expected ≈1.0)", prob_sum
            )

        return {
            "model":              self.model_name,
            "node_id":            input_data["node_id"],
            "timestamp":          input_data["timestamp"],
            "top_prediction":     top_prediction,
            "top_confidence":     top_confidence,
            "class_probabilities": class_probs,
            "inference_time_ms":  round(inference_ms, 2),
        }
