"""MaxViT extravasation head for the 1 mm, 5-slice checkpoints.

The encoder still sees one 3-channel difference image. The checkpoints
run that encoder on five views (offsets −2, −1, 0, +1, +2 mm) and combine
the five class-1 logits with a learned Linear(5, 1). ``forward`` returns
the sigmoid of the fused logit.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

N_VIEWS = 5


def _gem(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class BleedModel(nn.Module):
    def __init__(self, n_views: int = N_VIEWS):
        super().__init__()
        self.n_views = int(n_views)
        self.encoder = timm.create_model(
            "maxvit_tiny_tf_512",
            pretrained=False,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.1,
        )
        self.dropout = nn.Dropout(0.1)
        self.logits = nn.Linear(self.encoder.num_features, 11)
        self.fuse = nn.Linear(self.n_views, 1, bias=True)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        with torch.no_grad():
            self.fuse.weight[0, self.n_views // 2] = 1.0

    def view_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Class-1 logit for each of the ``n_views`` stacked difference images.

        ``x`` is (B, 3 * n_views, H, W), views packed as
        ncdiff(z−2) ‖ ncdiff(z−1) ‖ ncdiff(z) ‖ ncdiff(z+1) ‖ ncdiff(z+2).
        """
        b, c, h, w = x.shape
        n = self.n_views
        if c != 3 * n:
            raise ValueError(f"expected {3 * n} channels ({n} views), got {c}")
        flat = x.reshape(b * n, 3, h, w)
        features = _gem(self.encoder(flat))[:, :, 0, 0]
        return self.logits(self.dropout(features))[:, 1].view(b, n)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.view_logits(x)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(x))


def load_bleed_model(path: str | Path, device: torch.device) -> BleedModel:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model = BleedModel()
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [
        k for k in missing
        if not k.startswith("fuse.") and not k.endswith("num_batches_tracked")
    ]
    if bad or unexpected:
        raise RuntimeError(f"{path}: missing={bad[:8]} unexpected={list(unexpected)[:8]}")
    fuse = raw.get("fuse") if isinstance(raw, dict) else None
    if fuse is not None:
        model.fuse.load_state_dict(fuse)
    model.eval().to(device)
    return model
