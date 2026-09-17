"""Anchored radiometric nuisance views for the legacy 2D ResCUT discriminator."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_anchor(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "rescut_radiometric_anchor_v1":
        raise ValueError(f"Unsupported radiometric calibration: {path}")
    for domain in ("source", "target"):
        affine = payload.get("affine", {}).get(domain, {})
        if float(affine.get("scale", 0.0)) <= 0.0 or "shift" not in affine:
            raise ValueError(f"Invalid {domain} affine map in {path}")
    return payload


def body_mask(x: torch.Tensor) -> torch.Tensor:
    """The legacy data maps the air background exactly to -1."""
    return (x > -0.99).to(dtype=x.dtype)


def masked_affine(x: torch.Tensor, mask: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
    transformed = x * float(scale) + float(shift)
    return x + mask * (transformed - x)


def _shape(x: torch.Tensor) -> tuple[int, int, int, int]:
    return x.shape[0], x.shape[1], 1, 1


def _uniform(x: torch.Tensor, half_range: float) -> torch.Tensor:
    if half_range <= 0:
        return torch.zeros(_shape(x), device=x.device, dtype=x.dtype)
    return (2.0 * torch.rand(_shape(x), device=x.device, dtype=x.dtype) - 1.0) * float(half_range)


def sample_brion(x: torch.Tensor, scale_range: float, shift_range: float, piecewise_range: float) -> dict:
    base_scale = 1.0 + _uniform(x, scale_range)
    slope_delta = _uniform(x, piecewise_range)
    return {
        "slope_low": (base_scale * (1.0 - slope_delta)).clamp_min(0.1),
        "slope_high": (base_scale * (1.0 + slope_delta)).clamp_min(0.1),
        "shift": _uniform(x, shift_range),
    }


def apply_brion(x: torch.Tensor, mask: torch.Tensor, params: dict) -> torch.Tensor:
    transformed = torch.where(
        x <= 0.0,
        x * params["slope_low"] + params["shift"],
        x * params["slope_high"] + params["shift"],
    )
    return x + mask * (transformed - x)


def sample_fan_shading(x: torch.Tensor, mask: torch.Tensor, strength: float) -> torch.Tensor:
    """Body-centred, low-dimensional smooth shading field for 2D slices."""
    if strength <= 0:
        return torch.zeros_like(x)
    _, _, height, width = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    xx, yy = xx[None, None], yy[None, None]
    weight = F.avg_pool2d(mask, kernel_size=9, stride=1, padding=4) * mask
    denom = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    center_x = (weight * xx).sum(dim=(2, 3), keepdim=True) / denom
    center_y = (weight * yy).sum(dim=(2, 3), keepdim=True) / denom
    spread_x = ((weight * (xx - center_x).square()).sum(dim=(2, 3), keepdim=True) / denom).sqrt().clamp_min(0.15)
    spread_y = ((weight * (yy - center_y).square()).sum(dim=(2, 3), keepdim=True) / denom).sqrt().clamp_min(0.15)
    radial = ((xx - center_x) / (2.0 * spread_x)).square() + ((yy - center_y) / (2.0 * spread_y)).square()
    coeff_1 = _uniform(x, 1.0)
    coeff_2 = _uniform(x, 1.0)
    field = coeff_1 * radial + coeff_2 * radial.square()
    field = field - (field * weight).sum(dim=(2, 3), keepdim=True) / denom
    rms = ((field.square() * weight).sum(dim=(2, 3), keepdim=True) / denom).sqrt().clamp_min(1e-6)
    amplitude = torch.rand(_shape(x), device=x.device, dtype=x.dtype) * float(strength)
    return (field / rms * amplitude * weight).clamp(-2.0 * float(strength), 2.0 * float(strength))
