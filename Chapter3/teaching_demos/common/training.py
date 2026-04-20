"""Manifold-aware denoising-score-matching loss for diffusion teaching notebooks."""
from __future__ import annotations

from typing import Mapping

import torch

EPS = 1e-4


def _sigma_to_tensor(sigma) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        return sigma.detach().cpu().float()
    return torch.as_tensor(sigma, dtype=torch.float32)


def translation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma,
    *,
    divide_by_sigma2: bool = False,
) -> torch.Tensor:
    sigma_t = _sigma_to_tensor(sigma).to(pred.device)
    sq = (pred - target) ** 2
    if divide_by_sigma2:
        return (sq / (sigma_t ** 2).clamp_min(EPS)).mean()
    return (sq * sigma_t ** 2).mean()


def rotation_loss(pred: torch.Tensor, target: torch.Tensor, sigma, so3_module) -> torch.Tensor:
    norm = so3_module.score_norm(sigma).to(pred.device).clamp_min(EPS)
    return (((pred - target) / norm) ** 2).mean()


def torus_loss(pred: torch.Tensor, target: torch.Tensor, sigma, torus_module) -> torch.Tensor:
    sn = torch.as_tensor(torus_module.score_norm(sigma), dtype=pred.dtype, device=pred.device).clamp_min(EPS)
    return ((pred - target) ** 2 / sn).mean()


def l1_loss(pred: torch.Tensor, target: torch.Tensor, sigma) -> torch.Tensor:
    """Plain L1 loss (sigma unused). Matches raw DynamicBind res_tr/res_rot (training.py:128-131)."""
    return (pred - target).abs().mean()


def cosine_distance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma,
    *,
    symmetry_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cosine-distance loss `1 - cos(pred - target)`.

    Matches raw DynamicBind res_chi loss at utils/training.py:140-145.
    When ``symmetry_mask`` (bool, same shape as ``pred``) is provided, the
    C2-symmetric branch ``1 - cos(diff - pi)`` is min-pooled per element only
    where the mask is true, matching raw's `res_chi_symmetry_mask` semantics.
    Default: no symmetry (teaching uses CB-projection chi proxy, not real chi1
    with rotamer symmetry); avoids under-penalizing 180-degree errors on
    non-symmetric residues.
    """
    diff = pred - target
    l_primary = 1.0 - torch.cos(diff)
    if symmetry_mask is None:
        return l_primary.mean()
    l_symmetric = 1.0 - torch.cos(diff - torch.pi)
    sym = symmetry_mask.to(pred.device)
    return torch.where(sym, torch.minimum(l_primary, l_symmetric), l_primary).mean()


def manifold_score_loss(
    channels: Mapping[str, dict],
    *,
    so3_module=None,
    torus_module=None,
    weights: Mapping[str, float] | None = None,
) -> dict:
    if so3_module is None:
        from . import so3 as so3_module
    if torus_module is None:
        from . import torus as torus_module
    weights = weights or {}

    out: dict[str, torch.Tensor] = {}
    total = None
    for name, spec in channels.items():
        kind = spec["kind"]
        pred, target, sigma = spec["pred"], spec["target"], spec["sigma"]
        if kind == "translation":
            l = translation_loss(pred, target, sigma)
        elif kind == "rotation":
            l = rotation_loss(pred, target, sigma, so3_module)
        elif kind == "torus":
            l = torus_loss(pred, target, sigma, torus_module)
        elif kind == "l1":
            l = l1_loss(pred, target, sigma)
        elif kind == "cosine_distance":
            l = cosine_distance_loss(pred, target, sigma)
        else:
            raise ValueError(f"unknown channel kind={kind!r}")
        out[name] = l
        w = float(weights.get(name, 1.0))
        total = w * l if total is None else total + w * l
    out["total"] = total if total is not None else torch.zeros(1)
    return out
