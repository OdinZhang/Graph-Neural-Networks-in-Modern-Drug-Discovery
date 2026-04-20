"""Wrapped Gaussian (SO(2)) sampling and scoring for torsion / chi angles.

Ported from `Chapter3/docking/raw/flexible-docking/DiffDock-Pocket/utils/torus.py`
with reduced grid (SIGMA_N=1000, X_N=1000, N=50 wrapping terms vs raw's
5000/5000/100) for CPU-friendly teaching speed.

Cache written to ``common/_cache/`` on first use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

X_MIN, X_N = 1e-5, 1000          # relative to π
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2.0, 1000  # relative to π
N_WRAP = 50                       # number of wrap-around terms

_CACHE_DIR = Path(__file__).parent / "_cache"
_P_CACHE = _CACHE_DIR / "torus_p.npy"
_SCORE_CACHE = _CACHE_DIR / "torus_score.npy"

_x_grid = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
_sigma_grid = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi


def _p_grid(x: np.ndarray, sigma: np.ndarray, N: int = N_WRAP) -> np.ndarray:
    out = np.zeros_like(x + sigma)
    for i in range(-N, N + 1):
        out += np.exp(-((x + 2 * np.pi * i) ** 2) / (2 * sigma ** 2))
    return out


def _grad_grid(x: np.ndarray, sigma: np.ndarray, N: int = N_WRAP) -> np.ndarray:
    out = np.zeros_like(x + sigma)
    for i in range(-N, N + 1):
        out += (x + 2 * np.pi * i) / (sigma ** 2) * np.exp(
            -((x + 2 * np.pi * i) ** 2) / (2 * sigma ** 2)
        )
    return out


def _precompute_and_cache():
    _CACHE_DIR.mkdir(exist_ok=True)
    p_table = _p_grid(_x_grid, _sigma_grid[:, None])
    grad_table = _grad_grid(_x_grid, _sigma_grid[:, None])
    score_table = np.divide(
        grad_table, p_table, out=np.zeros_like(grad_table), where=p_table > 1e-30
    )
    np.save(_P_CACHE, p_table)
    np.save(_SCORE_CACHE, score_table)
    return p_table, score_table


if _P_CACHE.exists() and _SCORE_CACHE.exists():
    _p_table = np.load(_P_CACHE)
    _score_table = np.load(_SCORE_CACHE)
else:
    _p_table, _score_table = _precompute_and_cache()


def _x_to_idx(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    lx = np.log(np.abs(x) / np.pi + 1e-30)
    idx = (lx - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    idx = np.round(np.clip(idx, 0, X_N)).astype(int)
    return idx, sign


def _sigma_to_idx(sigma: np.ndarray) -> np.ndarray:
    ls = np.log(sigma / np.pi)
    idx = (ls - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    return np.round(np.clip(idx, 0, SIGMA_N)).astype(int)


def score(x, sigma) -> np.ndarray:
    """Wrapped-Gaussian score ∇log p(x|σ) for torus values."""
    x = np.asarray(x, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    idx_x, sign = _x_to_idx(x)
    idx_s = _sigma_to_idx(sigma)
    return -sign * _score_table[idx_s, idx_x]


def p(x, sigma) -> np.ndarray:
    """Wrapped-Gaussian density p(x|σ)."""
    x = np.asarray(x, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    idx_x, _ = _x_to_idx(x)
    idx_s = _sigma_to_idx(sigma)
    return _p_table[idx_s, idx_x]


def sample(sigma) -> np.ndarray:
    """Draw a wrapped-Gaussian sample at noise level *sigma*."""
    sigma = np.asarray(sigma, dtype=np.float64)
    out = sigma * np.random.randn(*sigma.shape)
    return (out + np.pi) % (2 * np.pi) - np.pi


_score_norm_cache = score(
    sample(_sigma_grid[None].repeat(2000, 0).flatten()),
    _sigma_grid[None].repeat(2000, 0).flatten(),
).reshape(2000, -1)
_score_norm_cache = (_score_norm_cache ** 2).mean(0)


def score_norm(sigma) -> np.ndarray:
    """Mean-squared score norm at noise level *sigma* (used to normalize loss)."""
    sigma = np.asarray(sigma, dtype=np.float64)
    idx_s = _sigma_to_idx(sigma)
    return _score_norm_cache[idx_s]
