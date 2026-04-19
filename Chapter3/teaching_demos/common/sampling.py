"""Generic reverse-diffusion sampling loop for the four pose-prediction notebooks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .diffusion_utils import (
    axis_angle_to_matrix,
    modify_torsion_angles,
    t_to_sigma_individual,
)


ApplyUpdate = Callable[..., None]


@dataclass
class Channel:
    """One diffusion channel.

    ``g_convention`` selects the diffusion-coefficient formula that matches the
    source raw repo:
      - "standard"      : g = sigma * sqrt(2 * log(sigma_max/sigma_min))
                          (raw DiffDock utils/sampling.py:134 for rotation,
                           and raw DiffDock-Pocket/DynamicBind for tr/tor)
      - "rot_scaled_2"  : g = 2 * sigma * sqrt(log(sigma_max/sigma_min))
                          (raw DiffDock-Pocket/DynamicBind utils/sampling.py:130
                           for rotation only)
      - "res_constant_3": g = 3 * sqrt(2 * log(sigma_max/sigma_min))
                          sigma-independent; matches raw DynamicBind residue
                          channels (utils/sampling.py:157-158)
    Default for rot/res_rot names is "rot_scaled_2"; for all other channels
    is "standard". Override per notebook as needed (e.g. DiffDock uses
    "standard" for its rotation channel; DynamicBind uses "res_constant_3"
    for residue channels).
    """

    name: str
    sigma_min: float
    sigma_max: float
    schedule_type: str = "exponential"
    score_key: str | None = None
    g_convention: str | None = None

    def __post_init__(self):
        if self.score_key is None:
            self.score_key = self.name
        if self.g_convention is None:
            self.g_convention = "rot_scaled_2" if self.name in {"rot", "res_rot"} else "standard"


def _haar_rotation_matrix() -> np.ndarray:
    """Uniform-Haar rotation matrix on SO(3), via scipy.spatial.transform.Rotation.

    Matches raw DiffDock/DiffDock-Pocket/DynamicBind ``randomize_position``
    which uses ``Rotation.random().as_matrix()`` (uniform SO(3)), NOT IGSO(3)
    sampled at sigma_rot_max.
    """
    from scipy.spatial.transform import Rotation as _R
    return _R.random().as_matrix().astype(np.float32)


def randomize_position_3way(
    coords: np.ndarray,
    *,
    sigma_tr_max: float,
    so3_module=None,
    sigma_rot_max: float | None = None,  # kept for API stability; ignored for init rotation
    mol=None,
    rot_bonds: Sequence[tuple[int, int]] | None = None,
    sigma_tor_max: float | None = None,
) -> np.ndarray:
    """Init pose: uniform torsions + Haar-random rotation around centroid + Gaussian translation.

    Matches raw DiffDock ``utils/sampling.py:randomize_position`` (Haar rotation,
    not IGSO(3)). ``sigma_rot_max`` argument is accepted but unused — the init
    rotation is always uniform-Haar on SO(3) per raw semantics.
    """
    del sigma_rot_max  # raw uses Haar, not IGSO(3) at sigma_rot_max
    out = coords.astype(np.float32).copy()
    if mol is not None and rot_bonds is not None and sigma_tor_max is not None and rot_bonds:
        deltas = np.random.uniform(-np.pi, np.pi, size=len(rot_bonds)).astype(np.float32)
        out = modify_torsion_angles(out, mol, rot_bonds, deltas)
    centroid = out.mean(axis=0, keepdims=True)
    R = _haar_rotation_matrix()
    out = (out - centroid) @ R.T + centroid
    out += np.random.randn(1, 3).astype(np.float32) * float(sigma_tr_max)
    return out


def randomize_position_pocket(
    lig_coords: np.ndarray,
    pocket_ca_coords: np.ndarray,
    *,
    sigma_tr_max: float,
    so3_module=None,
    sigma_rot_max: float | None = None,  # kept for API stability; ignored
    mol=None,
    rot_bonds: Sequence[tuple[int, int]] | None = None,
    sigma_tor_max: float | None = None,
) -> np.ndarray:
    """Pocket-aware init: recenter ligand on pocket centroid + Haar rotation + Gaussian translation.

    Matches raw DiffDock-Pocket / DynamicBind ``utils/sampling.py``
    ``randomize_position`` ``pocket_knowledge=True`` branch (raw lines 30-60).
    ``sigma_rot_max`` accepted but unused.
    """
    del sigma_rot_max
    out = lig_coords.astype(np.float32).copy()
    if mol is not None and rot_bonds is not None and sigma_tor_max is not None and rot_bonds:
        deltas = np.random.uniform(-np.pi, np.pi, size=len(rot_bonds)).astype(np.float32)
        out = modify_torsion_angles(out, mol, rot_bonds, deltas)
    pocket_center = pocket_ca_coords.mean(axis=0, keepdims=True).astype(np.float32)
    centroid = out.mean(axis=0, keepdims=True)
    R = _haar_rotation_matrix()
    out = (out - centroid) @ R.T + pocket_center
    out += np.random.randn(1, 3).astype(np.float32) * float(sigma_tr_max)
    return out


def _expbeta_schedule(n_steps: int, alpha: float = 2.0, beta: float = 0.03) -> np.ndarray:
    """expbeta t-schedule matching raw DiffDock ``utils/sampling.get_t_schedule``.

    Returns n_steps+1 values from 1.0 to 0.0, with exp-decay concentration near 0
    when alpha>1 (spending more inference budget on low-sigma refinement).
    """
    x = np.linspace(0.0, 1.0, n_steps + 1)
    t = 1.0 - (np.exp(alpha * x) - 1.0) / (np.exp(alpha) - 1.0) * (1.0 - beta) - beta * x
    return t


def reverse_diffusion_loop(
    *,
    channels: Sequence[Channel],
    score_fn: Callable[[dict, dict, float], dict],
    apply_updates: dict[str, ApplyUpdate],
    state: dict,
    n_steps: int,
    use_sde: bool = False,
    no_final_step_noise: bool = True,
    rng: np.random.Generator | None = None,
    t_schedule_type: str = "linear",
    expbeta_alpha: float = 2.0,
    expbeta_beta: float = 0.03,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()

    if t_schedule_type == "expbeta":
        t_schedule = _expbeta_schedule(n_steps, alpha=expbeta_alpha, beta=expbeta_beta)
    else:
        t_schedule = np.linspace(1.0, 0.0, n_steps + 1)
    # Per-channel diffusion coefficient. "standard" and "rot_scaled_2" store the
    # sigma-free *factor* (multiplied by sigma at step time); "res_constant_3"
    # stores the full sigma-independent g value.
    g_factors = {}
    sigma_free = {}  # True = g is sigma-independent; False = g = sigma * g_factor
    for c in channels:
        log_ratio = math.log(c.sigma_max / c.sigma_min)
        if c.g_convention == "rot_scaled_2":
            g_factors[c.name] = 2.0 * math.sqrt(log_ratio)
            sigma_free[c.name] = False
        elif c.g_convention == "standard":
            g_factors[c.name] = math.sqrt(2.0 * log_ratio)
            sigma_free[c.name] = False
        elif c.g_convention == "res_constant_3":
            g_factors[c.name] = 3.0 * math.sqrt(2.0 * log_ratio)
            sigma_free[c.name] = True
        else:
            raise ValueError(f"unknown g_convention={c.g_convention!r} for channel {c.name}")

    for step in range(n_steps):
        t = float(t_schedule[step])
        dt = float(t_schedule[step] - t_schedule[step + 1])

        # Expose step/n_steps in state so per-channel apply closures can use
        # raw empirical schedules (e.g. DynamicBind residue update:
        # ``score / (n_steps - step + n_steps * 0.25)`` at raw sampling.py:160-163).
        state["_step"] = step
        state["_n_steps"] = n_steps

        sigmas = {
            c.name: float(t_to_sigma_individual(t, c.schedule_type, c.sigma_min, c.sigma_max))
            for c in channels
        }
        scores = score_fn(state, sigmas, t)

        inject_noise = use_sde and not (no_final_step_noise and step == n_steps - 1)
        for c in channels:
            sigma = sigmas[c.name]
            g = g_factors[c.name] if sigma_free[c.name] else sigma * g_factors[c.name]
            score = scores[c.score_key]
            z = float(rng.standard_normal()) if inject_noise else 0.0
            state.setdefault("_z", {})[c.name] = z
            try:
                apply_updates[c.name](state, score, dt, sigma, g, z=z)
            except TypeError:
                # apply fn using legacy 5-arg signature (ODE only)
                apply_updates[c.name](state, score, dt, sigma, g)

    state["_meta"] = {
        "channels": [c.name for c in channels],
        "n_steps": n_steps,
        "use_sde": use_sde,
        "t_schedule_type": t_schedule_type,
    }
    return state
