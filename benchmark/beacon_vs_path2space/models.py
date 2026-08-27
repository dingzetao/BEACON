"""Path2Space-B models: frozen CTransPath encoder + MLP abundance head."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbundanceMLP(nn.Module):
    """Path2Space-style MLP regressor (no graph). Output is non-negative on log1p scale."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: tuple[int, ...] = (512, 256, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x))


def _strip_prefix(state: dict, prefixes: tuple[str, ...]) -> dict:
    out = {}
    for k, v in state.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def load_ctranspath(weights_path: str | Path, device: torch.device) -> nn.Module:
    """
    Load frozen CTransPath (768-d).

    Tries, in order:
      1) path2space companion: path2space.frozen.ctrans.CTransPath
      2) local timm + ConvStem builder (benchmark/beacon_vs_path2space/ctrans)
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"CTransPath weights not found: {weights_path}\n"
            "Download ctranspath.pth from Zenodo "
            "https://doi.org/10.5281/zenodo.20174301 "
            "and set path2space.ctranspath_weights in the YAML."
        )

    model = None
    try:
        from path2space.frozen.ctrans import CTransPath  # type: ignore

        model = CTransPath(num_classes=0)
        print("Using path2space.frozen.ctrans.CTransPath")
    except Exception:
        from ctrans import build_ctranspath_with_timm

        # Prefers local Swin+ConvStem (Zenodo weight-compatible); timm is fallback.
        model = build_ctranspath_with_timm(num_classes=0)

    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    state = _strip_prefix(state, ("module.", "backbone.", "encoder."))

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): e.g. {missing[:3]}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): e.g. {unexpected[:3]}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(device)


class CTransPathFeatureExtractor(nn.Module):
    """Frozen CTransPath with ImageNet norm + resize to 224."""

    def __init__(self, weights_path: str | Path, device: torch.device):
        super().__init__()
        self.encoder = load_ctranspath(weights_path, device)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        feats = self.encoder(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim > 2:
            # some swin forwards return B,N,C
            feats = feats.mean(dim=1)
        return feats.float()
