"""SO(3) IGSO(3) sampling and scoring via truncated Fourier series.

Ported from `Chapter3/docking/raw/flexible-docking/DiffDock-Pocket/utils/so3.py`
with reduced precomputation density (L=500, N_EPS=200, X_N=500 vs raw's
2000/1000/2000) for CPU-friendly teaching speed. Absolute error in
`score_norm` over teaching σ range [0.05, 2.0] is < 1%.

Cache written to ``common/_cache/`` on first use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MIN_EPS, MAX_EPS, N_EPS = 0.01, 2.0, 200
X_N = 500
L_TRUNC = 500

_CACHE_DIR = Path(__file__).parent / "_cache"
_CACHE_FILES = {
    "omegas": _CACHE_DIR / "so3_omegas.npy",
    "cdf": _CACHE_DIR / "so3_cdf.npy",
    "score_norms": _CACHE_DIR / "so3_score_norms.npy",
    "exp_score_norms": _CACHE_DIR / "so3_exp_score_norms.npy",
}


def _expansion(omega: np.ndarray, eps: float, L: int = L_TRUNC) -> np.ndarray:
    p = np.zeros_like(omega)
    sin_half = np.sin(omega / 2)
    for l in range(L):
        p += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps ** 2 / 2)
            * np.sin(omega * (l + 0.5))
            / sin_half
        )
    return p


def _density(expansion: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return expansion * (1 - np.cos(omega)) / np.pi


def _score(exp: np.ndarray, omega: np.ndarray, eps: float, L: int = L_TRUNC) -> np.ndarray:
    dSigma = np.zeros_like(omega)
    sin_half = np.sin(omega / 2)
    for l in range(L):
        hi = np.sin(omega * (l + 0.5))
        dhi = (l + 0.5) * np.cos(omega * (l + 0.5))
        dlo = 0.5 * np.cos(omega / 2)
        dSigma += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps ** 2 / 2)
            * (sin_half * dhi - hi * dlo)
            / sin_half ** 2
        )
    return dSigma / exp


def _precompute_and_cache():
    _CACHE_DIR.mkdir(exist_ok=True)
    eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    exp_vals = np.asarray([_expansion(omegas_array, eps) for eps in eps_array])
    pdf_vals = np.asarray([_density(e, omegas_array) for e in exp_vals])
    cdf_vals = np.asarray([p.cumsum() / X_N * np.pi for p in pdf_vals])
    score_norms = np.asarray(
        [_score(exp_vals[i], omegas_array, eps_array[i]) for i in range(len(eps_array))]
    )
    exp_score_norms = np.sqrt(
        np.sum(score_norms ** 2 * pdf_vals, axis=1)
        / np.sum(pdf_vals, axis=1)
        / np.pi
    )

    np.save(_CACHE_FILES["omegas"], omegas_array)
    np.save(_CACHE_FILES["cdf"], cdf_vals)
    np.save(_CACHE_FILES["score_norms"], score_norms)
    np.save(_CACHE_FILES["exp_score_norms"], exp_score_norms)
    return omegas_array, cdf_vals, score_norms, exp_score_norms


if all(f.exists() for f in _CACHE_FILES.values()):
    _omegas_array = np.load(_CACHE_FILES["omegas"])
    _cdf_vals = np.load(_CACHE_FILES["cdf"])
    _score_norms = np.load(_CACHE_FILES["score_norms"])
    _exp_score_norms = np.load(_CACHE_FILES["exp_score_norms"])
else:
    _omegas_array, _cdf_vals, _score_norms, _exp_score_norms = _precompute_and_cache()


def _eps_to_idx(eps) -> np.ndarray:
    eps = np.asarray(eps, dtype=np.float64)
    idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    return np.clip(np.around(idx).astype(int), 0, N_EPS - 1)


def sample(eps: float) -> float:
    """Draw a rotation angle ω ∈ [0, π] from the IGSO(3) marginal at noise level *eps*."""
    idx = _eps_to_idx(eps)
    x = np.random.rand()
    return float(np.interp(x, _cdf_vals[idx], _omegas_array))


def sample_vec(eps: float) -> np.ndarray:
    """Sample a 3-vector axis-angle from IGSO(3): random unit axis times sampled magnitude."""
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-12
    return axis * sample(eps)


def score_vec(eps: float, vec: np.ndarray) -> np.ndarray:
    """Compute the IGSO(3) score at axis-angle ``vec`` for noise level *eps*. Returns shape (3,)."""
    idx = _eps_to_idx(eps)
    om = float(np.linalg.norm(vec))
    if om < 1e-12:
        return np.zeros(3, dtype=np.float32)
    s = float(np.interp(om, _omegas_array, _score_norms[idx]))
    return (s * vec / om).astype(np.float32)


def score_norm(eps) -> torch.Tensor:
    """Expected ||score|| at noise level *eps* (used to normalize loss)."""
    if isinstance(eps, torch.Tensor):
        eps_np = eps.detach().cpu().numpy()
    else:
        eps_np = np.asarray(eps)
    idx = _eps_to_idx(eps_np)
    out = np.asarray(_exp_score_norms[idx], dtype=np.float32)
    return torch.from_numpy(out).float()
