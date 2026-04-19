"""Diffusion noise schedules and SE(3) coordinate operations."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def sigmoid_schedule(t, k: float = 10.0, m: float = 0.4):
    s = lambda u: sigmoid(k * (u - m))
    return (s(t) - s(0.0)) / (s(1.0) - s(0.0))


def t_to_sigma_individual(
    t,
    schedule_type: str,
    sigma_min: float,
    sigma_max: float,
    *,
    schedule_k: float = 10.0,
    schedule_m: float = 0.4,
):
    if schedule_type == "exponential":
        return sigma_min ** (1 - t) * sigma_max ** t
    if schedule_type == "sigmoid":
        return sigmoid_schedule(t, k=schedule_k, m=schedule_m) * (sigma_max - sigma_min) + sigma_min
    raise ValueError(f"unknown schedule_type={schedule_type}")


def t_to_sigma_3way(t, *, tr_range, rot_range, tor_range):
    return (
        t_to_sigma_individual(t, "exponential", *tr_range),
        t_to_sigma_individual(t, "exponential", *rot_range),
        t_to_sigma_individual(t, "exponential", *tor_range),
    )


def t_to_sigma_4way(t, *, tr_range, rot_range, tor_range, sc_range):
    return (
        t_to_sigma_individual(t, "exponential", *tr_range),
        t_to_sigma_individual(t, "exponential", *rot_range),
        t_to_sigma_individual(t, "exponential", *tor_range),
        t_to_sigma_individual(t, "exponential", *sc_range),
    )


def t_to_sigma_6way(t, *, tr_range, rot_range, tor_range, res_tr_range, res_rot_range, res_chi_range):
    return (
        t_to_sigma_individual(t, "exponential", *tr_range),
        t_to_sigma_individual(t, "exponential", *rot_range),
        t_to_sigma_individual(t, "exponential", *tor_range),
        t_to_sigma_individual(t, "exponential", *res_tr_range),
        t_to_sigma_individual(t, "exponential", *res_rot_range),
        t_to_sigma_individual(t, "exponential", *res_chi_range),
    )


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    v = np.asarray(axis_angle, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = v / angle
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)).astype(np.float32)


def axis_angle_to_matrix_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    v = axis_angle.reshape(3)
    angle = torch.linalg.norm(v)
    if angle < 1e-8:
        return torch.eye(3, dtype=v.dtype, device=v.device)
    axis = v / angle
    K = torch.zeros(3, 3, dtype=v.dtype, device=v.device)
    K[0, 1] = -axis[2]; K[0, 2] = axis[1]
    K[1, 0] = axis[2];  K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]; K[2, 1] = axis[0]
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    return eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def rodrigues_rotation(rot_vec: torch.Tensor) -> torch.Tensor:
    if rot_vec.dim() == 1:
        return axis_angle_to_matrix_torch(rot_vec).unsqueeze(0)
    angles = torch.linalg.norm(rot_vec, dim=-1, keepdim=True).clamp_min(1e-12)
    axes = rot_vec / angles
    angles = angles.squeeze(-1)
    sin_a = torch.sin(angles)
    cos_a = torch.cos(angles)
    one_minus_cos = 1 - cos_a
    N = rot_vec.shape[0]
    eye = torch.eye(3, dtype=rot_vec.dtype, device=rot_vec.device).expand(N, 3, 3)
    K = torch.zeros(N, 3, 3, dtype=rot_vec.dtype, device=rot_vec.device)
    K[:, 0, 1] = -axes[:, 2]; K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2];  K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]; K[:, 2, 1] = axes[:, 0]
    return eye + sin_a[:, None, None] * K + one_minus_cos[:, None, None] * torch.bmm(K, K)


def apply_translation_noise(coords: np.ndarray, sigma_tr: float):
    noise = np.random.randn(3).astype(np.float32) * float(sigma_tr)
    noisy = coords + noise
    target = -noise / max(float(sigma_tr) ** 2, 1e-8)
    return noisy, noise, target


def get_rotatable_bonds(mol):
    rot_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType().name != "SINGLE" or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(a).GetDegree() < 2 or mol.GetAtomWithIdx(b).GetDegree() < 2:
            continue
        rot_bonds.append((a, b))
    return rot_bonds


def _bfs_side(mol, start: int, blocked: int):
    visited = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j == blocked or j in visited:
                continue
            visited.add(j)
            queue.append(j)
    visited.discard(start)
    return visited


def _kabsch_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``source`` rigidly aligned onto ``target`` (Kabsch).

    Both arrays (N, 3). Result keeps source's internal geometry but matches
    target's rigid pose. Mirrors raw DiffDock
    ``utils/geometry.rigid_transform_Kabsch_3D_torch``.
    """
    src_c = source.mean(0, keepdims=True)
    tgt_c = target.mean(0, keepdims=True)
    H = (source - src_c).T @ (target - tgt_c)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d]).astype(source.dtype)
    R = Vt.T @ D @ U.T
    return (source - src_c) @ R.T + tgt_c


def modify_torsion_angles(
    coords: np.ndarray,
    mol,
    rot_bonds,
    delta_angles,
    *,
    kabsch_align: bool = True,
):
    """Rotate the 'other side' of each rotatable bond by ``delta_angles``.

    ``kabsch_align=True`` (default) post-aligns the rotated pose back to the
    pre-torsion rigid pose so torsion updates do not contaminate the tr/rot
    channels. Matches raw DiffDock ``diffusion_utils.py:41-55``.
    """
    new_coords = coords.copy().astype(np.float32)
    pre = new_coords.copy()
    for (a, b), delta in zip(rot_bonds, delta_angles):
        side = list(_bfs_side(mol, b, a))
        if not side:
            continue
        axis = new_coords[b] - new_coords[a]
        axis /= np.linalg.norm(axis) + 1e-12
        rot = axis_angle_to_matrix(axis * float(delta))
        pivot = new_coords[a]
        new_coords[side] = (new_coords[side] - pivot) @ rot.T + pivot
    if kabsch_align:
        new_coords = _kabsch_align(new_coords, pre).astype(np.float32)
    return new_coords


def apply_se3_noise(coords: np.ndarray, sigma_tr: float, sigma_rot: float, so3_module=None):
    if so3_module is None:
        from . import so3 as so3_module
    centroid = coords.mean(axis=0, keepdims=True)
    rot_vec = so3_module.sample_vec(float(sigma_rot)).astype(np.float32)
    R = axis_angle_to_matrix(rot_vec)
    rotated = (coords - centroid) @ R.T + centroid
    tr_noise = np.random.randn(3).astype(np.float32) * float(sigma_tr)
    noisy = rotated + tr_noise
    return noisy.astype(np.float32), tr_noise, rot_vec
