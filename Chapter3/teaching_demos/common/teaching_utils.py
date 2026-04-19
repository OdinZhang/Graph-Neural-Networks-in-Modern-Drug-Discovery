from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELEMENT_LIST = ["C", "N", "O", "S", "F", "P", "Cl", "Br"]
ATOM_FEAT_DIM = 10  # 8 elements + 1 other + 1 aromatic

STANDARD_AAS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
RESIDUE_FEAT_DIM = 21  # 20 standard AA + 1 other

# ---------------------------------------------------------------------------
# Project & data helpers
# ---------------------------------------------------------------------------


def find_project_root(marker: str = "demo_data") -> Path:
    """Locate the project root by walking up from cwd until *marker* dir is found."""
    here = Path(".").resolve()
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Cannot find directory containing {marker}/")


def parse_coreset(path) -> dict[str, float]:
    """Parse ``CoreSet.dat`` and return ``{pdbid: logKa}``."""
    labels: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                labels[parts[0]] = float(parts[3])
    return labels


def parse_pdb_ids(path) -> list[str]:
    """Return the list of PDB IDs from ``CoreSet.dat``."""
    return list(parse_coreset(path).keys())


# ---------------------------------------------------------------------------
# Molecular I/O  (RDKit)
# ---------------------------------------------------------------------------


def load_mol(path: str, fmt: str):
    """Load a molecule with RDKit.  *fmt*: ``'mol2'``, ``'sdf'``, or ``'pdb'``."""
    if fmt == "mol2":
        return Chem.MolFromMol2File(path, sanitize=True)
    elif fmt == "sdf":
        supplier = Chem.SDMolSupplier(path, sanitize=True)
        return next(iter(supplier))
    elif fmt == "pdb":
        return Chem.MolFromPDBFile(path, removeHs=False, sanitize=True)
    raise ValueError(f"Unknown format: {fmt}")


def load_complex(pdbid: str, complex_dir) -> dict:
    """Load a protein-ligand complex and return raw RDKit mols + 3-D coords.

    Returns a dict with keys:
    ``prot_mol``, ``lig_mol``, ``prot_coords``, ``lig_coords``.
    """
    complex_dir = str(complex_dir)
    pocket_path = os.path.join(complex_dir, pdbid, f"{pdbid}_pocket.pdb")
    ligand_mol2 = os.path.join(complex_dir, pdbid, f"{pdbid}_ligand.mol2")
    ligand_sdf = os.path.join(complex_dir, pdbid, f"{pdbid}_ligand.sdf")

    # protein pocket
    prot_mol = Chem.MolFromPDBFile(pocket_path, removeHs=True, sanitize=False)
    Chem.SanitizeMol(prot_mol)

    # ligand (mol2 first, sdf fallback)
    lig_mol = Chem.MolFromMol2File(ligand_mol2, sanitize=False)
    if lig_mol is None:
        lig_mol = load_mol(ligand_sdf, "sdf")
    Chem.SanitizeMol(lig_mol)

    prot_mol = Chem.RemoveHs(prot_mol)
    lig_mol = Chem.RemoveHs(lig_mol)

    # 3-D coordinates
    prot_conf = prot_mol.GetConformer()
    lig_conf = lig_mol.GetConformer()
    prot_coords = np.array(
        [prot_conf.GetAtomPosition(i) for i in range(prot_mol.GetNumAtoms())],
        dtype=np.float32,
    )
    lig_coords = np.array(
        [lig_conf.GetAtomPosition(i) for i in range(lig_mol.GetNumAtoms())],
        dtype=np.float32,
    )

    return {
        "prot_mol": prot_mol,
        "lig_mol": lig_mol,
        "prot_coords": prot_coords,
        "lig_coords": lig_coords,
    }


# ---------------------------------------------------------------------------
# Featurisation — atom level  (10-dim)
# ---------------------------------------------------------------------------


def atom_features(atom) -> np.ndarray:
    """Unified 10-dim atom feature vector.

    * dims 0-7 : element one-hot  (C N O S F P Cl Br)
    * dim  8   : other element
    * dim  9   : is aromatic
    """
    feat = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
    symbol = atom.GetSymbol()
    if symbol in ELEMENT_LIST:
        feat[ELEMENT_LIST.index(symbol)] = 1.0
    else:
        feat[8] = 1.0
    feat[9] = float(atom.GetIsAromatic())
    return feat


# ---------------------------------------------------------------------------
# Featurisation — residue level  (21-dim)
# ---------------------------------------------------------------------------


def residue_features(resname: str) -> np.ndarray:
    """21-dim residue one-hot (20 standard AA + other)."""
    feat = np.zeros(RESIDUE_FEAT_DIM, dtype=np.float32)
    if resname in STANDARD_AAS:
        feat[STANDARD_AAS.index(resname)] = 1.0
    else:
        feat[-1] = 1.0
    return feat


def extract_residue_data(pocket_pdb: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract residue-level features and CA coordinates from a pocket PDB via BioPython.

    Returns ``(res_feats (N_r, 21), ca_coords (N_r, 3))`` or ``None``.
    """
    from Bio.PDB import PDBParser  # type: ignore[attr-defined]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pocket_pdb)

    res_feats_list: list[np.ndarray] = []
    ca_coords_list: list[np.ndarray] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " " or "CA" not in residue:
                    continue
                res_feats_list.append(residue_features(residue.get_resname().strip()))
                ca_coords_list.append(residue["CA"].get_vector().get_array())

    return (
        np.array(res_feats_list, dtype=np.float32),
        np.array(ca_coords_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_rmsd(coords1, coords2) -> float:
    """RMSD between two coordinate arrays of shape ``(N, 3)``."""
    c1 = np.asarray(coords1, dtype=np.float32)
    c2 = np.asarray(coords2, dtype=np.float32)
    return float(np.sqrt(((c1 - c2) ** 2).sum(axis=-1).mean()))


# ---------------------------------------------------------------------------
# NN building blocks
# ---------------------------------------------------------------------------


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int] | int,
    output_dim: int,
    *,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build a minimal Linear + ReLU MLP for teaching notebooks."""
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]

    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        is_last = idx == len(dims) - 2
        if not is_last:
            layers.append(nn.ReLU())
        elif final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion time steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        device = t.device
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, dtype=torch.float32, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def describe_model_parameters(model: nn.Module) -> pd.DataFrame:
    rows = [
        {
            "层名称": name,
            "形状": str(tuple(param.shape)),
            "参数量": int(param.numel()),
        }
        for name, param in model.named_parameters()
    ]
    rows.append(
        {
            "层名称": "总计",
            "形状": "-",
            "参数量": int(sum(param.numel() for param in model.parameters())),
        }
    )
    return pd.DataFrame(rows)


def history_frame(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
) -> pd.DataFrame:
    epochs = list(range(1, len(train_losses) + 1))
    data = {
        "Epoch": epochs,
        "Train Loss": [float(v) for v in train_losses],
    }
    if val_losses is not None:
        data["Val Loss"] = [float(v) for v in val_losses]
    return pd.DataFrame(data)


def metric_frame(rows: Iterable[tuple[str, str | float | int]]) -> pd.DataFrame:
    metrics = list(rows)
    return pd.DataFrame(
        {
            "指标": [name for name, _ in metrics],
            "值": [value for _, value in metrics],
        }
    )


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", linewidth=2)
    if val_losses is not None and len(val_losses) == len(train_losses):
        ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
