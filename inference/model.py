"""MaxViT extravasation head. Returns the sigmoid of class 1."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _gem(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class BleedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "maxvit_tiny_tf_512",
            pretrained=False,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.1,
        )
        self.dropout = nn.Dropout(0.1)
        self.logits = nn.Linear(self.encoder.num_features, 11)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = _gem(self.encoder(x))[:, :, 0, 0]
        return torch.sigmoid(self.logits(self.dropout(features))[:, 1])


def load_bleed_model(path: str | Path, device: torch.device) -> BleedModel:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model = BleedModel()
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.endswith("num_batches_tracked")]
    if bad or unexpected:
        raise RuntimeError(f"{path}: missing={bad[:8]} unexpected={list(unexpected)[:8]}")
    model.eval().to(device)
    return model
