from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem import Draw


RELATIVE_TUTORIAL_DIR = Path(
    "teaching_demos/2.single_step_retro_tutorial/2.3.semi-template/2.3.1.g2gs"
)

TORCHDRUG_SOURCE_MAP = [
    {
        "teaching_module": "one_hot / atom_*_features / bond_default_features",
        "torchdrug_source": "source_repos/torchdrug/torchdrug/data/feature.py",
        "original_symbol": "atom_center_identification / atom_synthon_completion / bond_default",
        "role": "原子与化学键特征编码",
    },
    {
        "teaching_module": "compute_reaction_difference / identify_reaction_center / extract_synthons",
        "torchdrug_source": "source_repos/torchdrug/torchdrug/datasets/uspto50k.py",
        "original_symbol": "_get_difference / _get_reaction_center / _get_synthon",
        "role": "USPTO50k 数据预处理与 synthon 构造",
    },
    {
        "teaching_module": "CenterIdentificationModel",
        "torchdrug_source": "source_repos/torchdrug/torchdrug/tasks/retrosynthesis.py",
        "original_symbol": "CenterIdentification",
        "role": "反应中心识别头",
    },
    {
        "teaching_module": "SynthonCompletionModel / oracle_completion_actions",
        "torchdrug_source": "source_repos/torchdrug/torchdrug/tasks/retrosynthesis.py",
        "original_symbol": "SynthonCompletion / Retrosynthesis",
        "role": "synthon completion 与整体推理管线",
    },
    {
        "teaching_module": "SimpleRGCNLayer / SimpleRGCNEncoder",
        "torchdrug_source": "source_repos/torchdrug/torchdrug/models/gcn.py",
        "original_symbol": "RelationalGraphConvolutionalNetwork",
        "role": "分子图编码器",
    },
]

ATOM_VOCAB = [
    "H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"
]
ATOM_VOCAB_INDEX = {atom: i for i, atom in enumerate(ATOM_VOCAB)}
DEGREE_VOCAB = list(range(7))
NUM_HS_VOCAB = list(range(7))
TOTAL_VALENCE_VOCAB = list(range(8))

BOND_TYPE_VOCAB = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_TYPE_INDEX = {bond_type: i for i, bond_type in enumerate(BOND_TYPE_VOCAB)}
BOND_TYPE_NAME = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
BOND_DIR_VOCAB = list(range(len(Chem.rdchem.BondDir.values)))
BOND_STEREO_VOCAB = list(range(len(Chem.rdchem.BondStereo.values)))


@dataclass
class ReactionExample:
    reaction_id: str
    reaction_class: int
    reaction_name: str
    rxn_smiles: str
    reactant_smiles: str
    product_smiles: str
    reactant_mol: Chem.Mol
    product_mol: Chem.Mol


@dataclass
class GraphTensor:
    node_feature: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_feature: torch.Tensor
    atom_map: torch.Tensor
    batch: torch.Tensor
    num_nodes_per_graph: torch.Tensor
    smiles: list[str]

    @property
    def num_node(self) -> int:
        return int(self.node_feature.shape[0])

    @property
    def num_edge(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_graph(self) -> int:
        return int(self.num_nodes_per_graph.shape[0])

    def to(self, device: str | torch.device) -> "GraphTensor":
        return GraphTensor(
            node_feature=self.node_feature.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            edge_feature=self.edge_feature.to(device),
            atom_map=self.atom_map.to(device),
            batch=self.batch.to(device),
            num_nodes_per_graph=self.num_nodes_per_graph.to(device),
            smiles=list(self.smiles),
        )


def find_project_root(start: str | Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    if here.is_file():
        here = here.parent
    for candidate in [here, *here.parents]:
        if (candidate / "teaching_demos").exists() and (candidate / "source_repos").exists():
            return candidate
    raise FileNotFoundError("Cannot locate project root from current path")


def tutorial_dir(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root else find_project_root()
    return root / RELATIVE_TUTORIAL_DIR


def demo_csv_path(project_root: str | Path | None = None) -> Path:
    return tutorial_dir(project_root) / "data" / "demo_reactions.csv"


def processed_dir(project_root: str | Path | None = None) -> Path:
    path = tutorial_dir(project_root) / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_mapping_frame() -> pd.DataFrame:
    return pd.DataFrame(TORCHDRUG_SOURCE_MAP)


def one_hot(value, vocab, allow_unknown: bool = False) -> list[int]:
    if isinstance(vocab, dict):
        index = vocab.get(value, -1)
        size = len(vocab)
    else:
        index = vocab.index(value) if value in vocab else -1
        size = len(vocab)

    if allow_unknown:
        feature = [0] * (size + 1)
        feature[index] = 1
    else:
        if index == -1:
            raise ValueError(f"Unknown value `{value}` for vocab `{vocab}`")
        feature = [0] * size
        feature[index] = 1
    return feature


def atom_center_identification_features(atom: Chem.Atom) -> list[int]:
    return (
        one_hot(atom.GetSymbol(), ATOM_VOCAB_INDEX, allow_unknown=True)
        + one_hot(atom.GetTotalNumHs(), NUM_HS_VOCAB)
        + one_hot(atom.GetTotalDegree(), DEGREE_VOCAB, allow_unknown=True)
        + one_hot(atom.GetTotalValence(), TOTAL_VALENCE_VOCAB)
        + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
    )


def atom_synthon_completion_features(atom: Chem.Atom) -> list[int]:
    return (
        one_hot(atom.GetSymbol(), ATOM_VOCAB_INDEX, allow_unknown=True)
        + one_hot(atom.GetTotalNumHs(), NUM_HS_VOCAB)
        + one_hot(atom.GetTotalDegree(), DEGREE_VOCAB, allow_unknown=True)
        + [
            int(atom.IsInRing()),
            int(atom.IsInRingSize(3)),
            int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)),
            int(atom.IsInRingSize(6)),
            int(
                atom.IsInRing()
                and not atom.IsInRingSize(3)
                and not atom.IsInRingSize(4)
                and not atom.IsInRingSize(5)
                and not atom.IsInRingSize(6)
            ),
        ]
    )


def bond_default_features(bond: Chem.Bond) -> list[int]:
    return (
        one_hot(bond.GetBondType(), BOND_TYPE_INDEX)
        + one_hot(int(bond.GetBondDir()), BOND_DIR_VOCAB)
        + one_hot(int(bond.GetStereo()), BOND_STEREO_VOCAB)
        + [int(bond.GetIsConjugated())]
    )


def feature_dimensions() -> dict[str, int]:
    methane = Chem.MolFromSmiles("C")
    ethanol = Chem.MolFromSmiles("CCO")
    return {
        "center_identification_atom_dim": len(atom_center_identification_features(methane.GetAtomWithIdx(0))),
        "synthon_completion_atom_dim": len(atom_synthon_completion_features(ethanol.GetAtomWithIdx(0))),
        "bond_dim": len(bond_default_features(ethanol.GetBondWithIdx(0))),
    }


def mapped_mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES `{smiles}`")
    return mol


def split_reaction_smiles(rxn_smiles: str) -> tuple[str, str]:
    reactant_smiles, _, product_smiles = rxn_smiles.split(">")
    return reactant_smiles, product_smiles


def load_demo_reactions(csv_path: str | Path | None = None) -> list[ReactionExample]:
    path = Path(csv_path) if csv_path else demo_csv_path()
    frame = pd.read_csv(path)
    examples = []
    for row in frame.to_dict("records"):
        reactant_smiles, product_smiles = split_reaction_smiles(row["rxn_smiles"])
        examples.append(
            ReactionExample(
                reaction_id=row["reaction_id"],
                reaction_class=int(row["class"]),
                reaction_name=row["reaction_name"],
                rxn_smiles=row["rxn_smiles"],
                reactant_smiles=reactant_smiles,
                product_smiles=product_smiles,
                reactant_mol=mapped_mol_from_smiles(reactant_smiles),
                product_mol=mapped_mol_from_smiles(product_smiles),
            )
        )
    return examples


def canonical_smiles_with_mapping(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def _atom_feature_fn(mode: str):
    if mode == "center_identification":
        return atom_center_identification_features
    if mode == "synthon_completion":
        return atom_synthon_completion_features
    raise ValueError(f"Unknown atom feature mode `{mode}`")


def molecule_to_graph_tensor(
    mol: Chem.Mol,
    atom_feature_mode: str = "center_identification",
    batch_id: int = 0,
) -> GraphTensor:
    atom_feature_fn = _atom_feature_fn(atom_feature_mode)

    node_feature = []
    atom_map = []
    for atom in mol.GetAtoms():
        node_feature.append(atom_feature_fn(atom))
        atom_map.append(atom.GetAtomMapNum())

    edge_index = []
    edge_type = []
    edge_feature = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_INDEX[bond.GetBondType()]
        feature = bond_default_features(bond)
        edge_index.append([begin, end])
        edge_index.append([end, begin])
        edge_type.extend([bond_type, bond_type])
        edge_feature.append(feature)
        edge_feature.append(feature)

    dims = feature_dimensions()
    bond_dim = dims["bond_dim"]
    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        edge_feature_tensor = torch.tensor(edge_feature, dtype=torch.float32)
    else:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_type_tensor = torch.zeros((0,), dtype=torch.long)
        edge_feature_tensor = torch.zeros((0, bond_dim), dtype=torch.float32)

    num_nodes = len(node_feature)
    return GraphTensor(
        node_feature=torch.tensor(node_feature, dtype=torch.float32),
        edge_index=edge_index_tensor,
        edge_type=edge_type_tensor,
        edge_feature=edge_feature_tensor,
        atom_map=torch.tensor(atom_map, dtype=torch.long),
        batch=torch.full((num_nodes,), batch_id, dtype=torch.long),
        num_nodes_per_graph=torch.tensor([num_nodes], dtype=torch.long),
        smiles=[canonical_smiles_with_mapping(mol)],
    )


def batch_graphs(graphs: Iterable[GraphTensor]) -> GraphTensor:
    graphs = list(graphs)
    if not graphs:
        raise ValueError("Cannot batch an empty graph list")

    node_features = []
    edge_indices = []
    edge_types = []
    edge_features = []
    atom_maps = []
    batches = []
    num_nodes_per_graph = []
    smiles = []
    node_offset = 0

    for batch_id, graph in enumerate(graphs):
        node_features.append(graph.node_feature)
        atom_maps.append(graph.atom_map)
        batches.append(torch.full((graph.num_node,), batch_id, dtype=torch.long))
        num_nodes_per_graph.append(graph.num_node)
        smiles.extend(graph.smiles)
        if graph.num_edge:
            edge_indices.append(graph.edge_index + node_offset)
            edge_types.append(graph.edge_type)
            edge_features.append(graph.edge_feature)
        node_offset += graph.num_node

    feature_dim = graphs[0].edge_feature.shape[-1] if graphs[0].edge_feature.ndim == 2 else 0
    return GraphTensor(
        node_feature=torch.cat(node_features, dim=0),
        edge_index=torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.cat(edge_types, dim=0) if edge_types else torch.zeros((0,), dtype=torch.long),
        edge_feature=(
            torch.cat(edge_features, dim=0)
            if edge_features
            else torch.zeros((0, feature_dim), dtype=torch.float32)
        ),
        atom_map=torch.cat(atom_maps, dim=0),
        batch=torch.cat(batches, dim=0),
        num_nodes_per_graph=torch.tensor(num_nodes_per_graph, dtype=torch.long),
        smiles=smiles,
    )


def segment_sum(x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = x.new_zeros((dim_size, x.shape[-1]))
    out.index_add_(0, index, x)
    return out


def segment_mean(x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = segment_sum(x, index, dim_size)
    count = x.new_zeros((dim_size, 1))
    ones = x.new_ones((x.shape[0], 1))
    count.index_add_(0, index, ones)
    return out / count.clamp(min=1)


def _atom_map_to_index(mol: Chem.Mol) -> dict[int, int]:
    mapping = {}
    for atom in mol.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map > 0:
            mapping[atom_map] = atom.GetIdx()
    return mapping


def _bond_signature_from_mol(mol: Chem.Mol) -> dict[tuple[int, int], dict]:
    atom_map = _atom_map_to_index(mol)
    bonds = {}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_map = mol.GetAtomWithIdx(begin).GetAtomMapNum()
        end_map = mol.GetAtomWithIdx(end).GetAtomMapNum()
        if begin_map <= 0 or end_map <= 0:
            continue
        key = tuple(sorted((begin_map, end_map)))
        bonds[key] = {
            "bond_type": BOND_TYPE_INDEX[bond.GetBondType()],
            "bond_name": BOND_TYPE_NAME[BOND_TYPE_INDEX[bond.GetBondType()]],
            "bond_index": bond.GetIdx(),
            "atom_indices": (
                atom_map[min(key)],
                atom_map[max(key)],
            ),
        }
    return bonds


def compute_reaction_difference(reactant_mol: Chem.Mol, product_mol: Chem.Mol) -> dict:
    reactant_bonds = _bond_signature_from_mol(reactant_mol)
    product_bonds = _bond_signature_from_mol(product_mol)
    reactant_map = _atom_map_to_index(reactant_mol)
    product_map = _atom_map_to_index(product_mol)

    added_bonds = []
    modified_bonds = []
    for bond_key, product_info in product_bonds.items():
        reactant_info = reactant_bonds.get(bond_key)
        if reactant_info is None:
            added_bonds.append(bond_key)
        elif reactant_info["bond_type"] != product_info["bond_type"]:
            modified_bonds.append(bond_key)

    hydrogen_changed_atoms = []
    shared_maps = sorted(set(product_map) & set(reactant_map))
    for atom_map in shared_maps:
        product_atom = product_mol.GetAtomWithIdx(product_map[atom_map])
        reactant_atom = reactant_mol.GetAtomWithIdx(reactant_map[atom_map])
        if product_atom.GetTotalNumHs() != reactant_atom.GetTotalNumHs():
            hydrogen_changed_atoms.append(atom_map)

    return {
        "reactant_bonds": reactant_bonds,
        "product_bonds": product_bonds,
        "added_bonds": added_bonds,
        "modified_bonds": modified_bonds,
        "hydrogen_changed_atoms": hydrogen_changed_atoms,
    }


def identify_reaction_center(reactant_mol: Chem.Mol, product_mol: Chem.Mol) -> dict:
    difference = compute_reaction_difference(reactant_mol, product_mol)
    node_label = torch.zeros(product_mol.GetNumAtoms(), dtype=torch.long)
    edge_label = torch.zeros(product_mol.GetNumBonds() * 2, dtype=torch.long)
    product_map = _atom_map_to_index(product_mol)

    if len(difference["added_bonds"]) == 1:
        bond_key = difference["added_bonds"][0]
        bond_index = difference["product_bonds"][bond_key]["bond_index"]
        edge_label[2 * bond_index: 2 * bond_index + 2] = 1
        center = {
            "center_type": "bond",
            "reaction_center": bond_key,
            "center_atom_maps": bond_key,
            "valid": True,
        }
    elif len(difference["modified_bonds"]) == 1:
        atom_map_1, atom_map_2 = difference["modified_bonds"][0]
        atom_1 = product_mol.GetAtomWithIdx(product_map[atom_map_1])
        atom_2 = product_mol.GetAtomWithIdx(product_map[atom_map_2])
        if atom_1.GetDegree() == 1:
            chosen_map = atom_map_1
        elif atom_2.GetDegree() == 1:
            chosen_map = atom_map_2
        else:
            chosen_map = atom_map_1
        node_label[product_map[chosen_map]] = 1
        center = {
            "center_type": "atom",
            "reaction_center": (chosen_map, 0),
            "center_atom_maps": (chosen_map,),
            "valid": True,
        }
    elif len(difference["hydrogen_changed_atoms"]) == 1:
        chosen_map = difference["hydrogen_changed_atoms"][0]
        node_label[product_map[chosen_map]] = 1
        center = {
            "center_type": "atom",
            "reaction_center": (chosen_map, 0),
            "center_atom_maps": (chosen_map,),
            "valid": True,
        }
    else:
        center = {
            "center_type": "invalid",
            "reaction_center": tuple(),
            "center_atom_maps": tuple(),
            "valid": False,
        }

    center.update(
        {
            "node_label": node_label,
            "edge_label": edge_label,
            "difference": difference,
        }
    )
    return center


def _sanitize_fragment(fragment: Chem.Mol) -> Chem.Mol:
    smiles = Chem.MolToSmiles(fragment, canonical=True)
    return Chem.MolFromSmiles(smiles)


def split_molecule_components(mol: Chem.Mol) -> list[Chem.Mol]:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    sanitized = []
    for fragment in fragments:
        sanitized.append(_sanitize_fragment(fragment))
    return sanitized


def _mapped_atom_set(mol: Chem.Mol) -> set[int]:
    return {atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}


def _remove_bond_by_map_numbers(product_mol: Chem.Mol, atom_map_1: int, atom_map_2: int) -> Chem.Mol:
    mapping = _atom_map_to_index(product_mol)
    editable = Chem.RWMol(Chem.Mol(product_mol))
    editable.RemoveBond(mapping[atom_map_1], mapping[atom_map_2])
    return editable.GetMol()


def match_reactants_and_synthons(reactant_mol: Chem.Mol, synthons: list[Chem.Mol]) -> list[tuple[Chem.Mol, Chem.Mol]]:
    reactant_components = split_molecule_components(reactant_mol)
    synthon_sets = [_mapped_atom_set(mol) for mol in synthons]
    used = set()
    pairs = []
    for reactant_component in reactant_components:
        reactant_set = _mapped_atom_set(reactant_component)
        best_idx = None
        best_overlap = -1
        for idx, synthon_set in enumerate(synthon_sets):
            if idx in used:
                continue
            overlap = len(reactant_set & synthon_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        if best_idx is not None and best_overlap > 0:
            used.add(best_idx)
            pairs.append((reactant_component, synthons[best_idx]))
    return pairs


def extract_synthons(
    reactant_mol: Chem.Mol,
    product_mol: Chem.Mol,
    center: dict | None = None,
) -> list[tuple[Chem.Mol, Chem.Mol]]:
    center = center or identify_reaction_center(reactant_mol, product_mol)
    if not center["valid"]:
        return []
    if center["center_type"] == "bond":
        atom_map_1, atom_map_2 = center["reaction_center"]
        disconnected = _remove_bond_by_map_numbers(product_mol, atom_map_1, atom_map_2)
        synthons = split_molecule_components(disconnected)
        return match_reactants_and_synthons(reactant_mol, synthons)
    return [(Chem.Mol(reactant_mol), Chem.Mol(product_mol))]


def reaction_center_mask(graph: GraphTensor, reaction_center: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros(graph.num_node, dtype=torch.bool)
    if reaction_center:
        selected = set(reaction_center)
        for idx, atom_map in enumerate(graph.atom_map.tolist()):
            if atom_map in selected:
                mask[idx] = True
    return mask


def oracle_completion_actions(reactant_mol: Chem.Mol, synthon_mol: Chem.Mol) -> list[dict]:
    reactant_bonds = _bond_signature_from_mol(reactant_mol)
    synthon_bonds = _bond_signature_from_mol(synthon_mol)
    synthon_atom_set = _mapped_atom_set(synthon_mol)
    actions = []

    for bond_key, reactant_info in reactant_bonds.items():
        synthon_info = synthon_bonds.get(bond_key)
        if synthon_info is None:
            atom_map_1, atom_map_2 = bond_key
            if atom_map_1 in synthon_atom_set and atom_map_2 in synthon_atom_set:
                actions.append(
                    {
                        "action_type": "add_bond",
                        "node_in_map": atom_map_1,
                        "node_out_map": atom_map_2,
                        "bond_type": reactant_info["bond_name"],
                    }
                )
            else:
                actions.append(
                    {
                        "action_type": "new_atom_or_leaving_group",
                        "node_in_map": atom_map_1,
                        "node_out_map": atom_map_2,
                        "bond_type": reactant_info["bond_name"],
                    }
                )
        elif synthon_info["bond_type"] != reactant_info["bond_type"]:
            atom_map_1, atom_map_2 = bond_key
            actions.append(
                {
                    "action_type": "change_bond",
                    "node_in_map": atom_map_1,
                    "node_out_map": atom_map_2,
                    "bond_type": reactant_info["bond_name"],
                }
            )

    if not actions:
        difference = compute_reaction_difference(reactant_mol, synthon_mol)
        for atom_map in difference["hydrogen_changed_atoms"]:
            actions.append(
                {
                    "action_type": "atom_property_change",
                    "node_in_map": atom_map,
                    "node_out_map": None,
                    "bond_type": None,
                }
            )

    actions.append(
        {
            "action_type": "stop",
            "node_in_map": None,
            "node_out_map": None,
            "bond_type": None,
        }
    )
    return actions


def build_center_identification_dataset(examples: Iterable[ReactionExample]) -> list[dict]:
    dataset = []
    for example in examples:
        center = identify_reaction_center(example.reactant_mol, example.product_mol)
        dataset.append(
            {
                "reaction_id": example.reaction_id,
                "reaction_class": example.reaction_class,
                "reaction_name": example.reaction_name,
                "reactant_graph": molecule_to_graph_tensor(example.reactant_mol, "center_identification"),
                "product_graph": molecule_to_graph_tensor(example.product_mol, "center_identification"),
                "center_target": center,
                "rxn_smiles": example.rxn_smiles,
            }
        )
    return dataset


def build_synthon_completion_dataset(examples: Iterable[ReactionExample]) -> list[dict]:
    dataset = []
    for example in examples:
        center = identify_reaction_center(example.reactant_mol, example.product_mol)
        for pair_index, (reactant_mol, synthon_mol) in enumerate(
            extract_synthons(example.reactant_mol, example.product_mol, center)
        ):
            dataset.append(
                {
                    "reaction_id": example.reaction_id,
                    "pair_id": f"{example.reaction_id}_pair_{pair_index}",
                    "reaction_class": example.reaction_class,
                    "reaction_name": example.reaction_name,
                    "reaction_center": center["reaction_center"],
                    "reactant_graph": molecule_to_graph_tensor(reactant_mol, "synthon_completion"),
                    "synthon_graph": molecule_to_graph_tensor(synthon_mol, "synthon_completion"),
                    "oracle_actions": oracle_completion_actions(reactant_mol, synthon_mol),
                    "reactant_smiles": canonical_smiles_with_mapping(reactant_mol),
                    "synthon_smiles": canonical_smiles_with_mapping(synthon_mol),
                }
            )
    return dataset


def export_processed_demo(
    center_dataset: list[dict],
    synthon_dataset: list[dict],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    target_dir = Path(output_dir) if output_dir else processed_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    center_path = target_dir / "center_identification_demo.pkl"
    synthon_path = target_dir / "synthon_completion_demo.pkl"
    with center_path.open("wb") as fout:
        pickle.dump(center_dataset, fout)
    with synthon_path.open("wb") as fout:
        pickle.dump(synthon_dataset, fout)
    return {"center": center_path, "synthon": synthon_path}


def _normalize_svg_markup(svg: str) -> str:
    svg = str(svg).strip()
    if svg.startswith("<?xml"):
        _, _, remainder = svg.partition("?>")
        svg = remainder.strip()
    start = svg.find("<svg")
    end = svg.rfind("</svg>")
    if start != -1 and end != -1:
        svg = svg[start: end + len("</svg>")]
    return svg


def svg_html(svg: str):
    from IPython.display import HTML

    return HTML(_normalize_svg_markup(svg))


def _prepare_mol_for_drawing(mol: Chem.Mol) -> Chem.Mol:
    safe = Chem.Mol(mol)
    for atom in safe.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(safe, canonical=True, isomericSmiles=True)
    redrawn = Chem.MolFromSmiles(smiles)
    return redrawn if redrawn is not None else safe


def draw_molecule_svg(mol: Chem.Mol, legends: list[str] | None = None, mols_per_row: int = 1) -> str:
    mols = [mol] if not isinstance(mol, list) else mol
    mols = [_prepare_mol_for_drawing(item) for item in mols]
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(350, 260),
        legends=legends,
        useSVG=True,
    )
    return _normalize_svg_markup(str(image))


def draw_molecule_image(mol: Chem.Mol, legends: list[str] | None = None, mols_per_row: int = 1):
    mols = [mol] if not isinstance(mol, list) else mol
    mols = [_prepare_mol_for_drawing(item) for item in mols]
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(350, 260),
        legends=legends,
        useSVG=False,
    )
    return image


def draw_reaction_pair_svg(reactant_mol: Chem.Mol, product_mol: Chem.Mol, legends: list[str] | None = None) -> str:
    legends = legends or ["Reactant", "Product"]
    mols = [_prepare_mol_for_drawing(reactant_mol), _prepare_mol_for_drawing(product_mol)]
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=(350, 260),
        legends=legends,
        useSVG=True,
    )
    return _normalize_svg_markup(str(image))


def draw_reaction_pair_image(reactant_mol: Chem.Mol, product_mol: Chem.Mol, legends: list[str] | None = None):
    legends = legends or ["Reactant", "Product"]
    mols = [_prepare_mol_for_drawing(reactant_mol), _prepare_mol_for_drawing(product_mol)]
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=(350, 260),
        legends=legends,
        useSVG=False,
    )
    return image


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimpleRGCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_relation: int):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.relation_linears = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=False) for _ in range(num_relation)]
        )
        self.output_dim = output_dim

    def forward(self, graph: GraphTensor, x: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        src, dst = graph.edge_index
        messages = x.new_zeros((graph.num_edge, self.output_dim))
        for relation_id, linear in enumerate(self.relation_linears):
            mask = graph.edge_type == relation_id
            if mask.any():
                messages[mask] = linear(x[src[mask]])
        aggregated = x.new_zeros((num_nodes, self.output_dim))
        if graph.num_edge:
            aggregated.index_add_(0, dst, messages)
            degree = torch.bincount(dst, minlength=num_nodes).to(x.dtype).unsqueeze(-1).clamp(min=1)
            aggregated = aggregated / degree
        return F.relu(self.self_linear(x) + aggregated)


class SimpleRGCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_relation: int, concat_hidden: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            self.layers.append(SimpleRGCNLayer(dims[i], dims[i + 1], num_relation))
        self.concat_hidden = concat_hidden
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]

    def forward(self, graph: GraphTensor, node_input: torch.Tensor) -> dict[str, torch.Tensor]:
        hiddens = []
        hidden = node_input
        for layer in self.layers:
            hidden = layer(graph, hidden)
            hiddens.append(hidden)
        node_feature = torch.cat(hiddens, dim=-1) if self.concat_hidden else hiddens[-1]
        graph_feature = segment_mean(node_feature, graph.batch, graph.num_graph)
        return {"node_feature": node_feature, "graph_feature": graph_feature}


def _reaction_one_hot(reaction: int | torch.Tensor | None, num_reaction: int, device, num_graph: int) -> torch.Tensor:
    if reaction is None:
        reaction = torch.zeros(num_graph, dtype=torch.long, device=device)
    elif isinstance(reaction, int):
        reaction = torch.tensor([reaction] * num_graph, dtype=torch.long, device=device)
    else:
        reaction = reaction.to(device)
        if reaction.ndim == 0:
            reaction = reaction.repeat(num_graph)
    one_hot = torch.zeros((num_graph, num_reaction), dtype=torch.float32, device=device)
    one_hot.scatter_(1, reaction.unsqueeze(-1), 1.0)
    return one_hot


class CenterIdentificationModel(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        num_relation: int,
        num_reaction: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_reaction = num_reaction
        self.encoder = SimpleRGCNEncoder(
            input_dim=node_input_dim,
            hidden_dims=[hidden_dim] * num_layers,
            num_relation=num_relation,
            concat_hidden=True,
        )
        graph_context_dim = self.encoder.output_dim + num_reaction
        node_context_dim = self.encoder.output_dim + node_input_dim + graph_context_dim
        edge_context_dim = edge_input_dim + node_context_dim * 2
        self.node_mlp = MLP(node_context_dim, hidden_dim, 1)
        self.edge_mlp = MLP(edge_context_dim, hidden_dim, 1)

    def forward(self, graph: GraphTensor, reaction: int | torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        output = self.encoder(graph, graph.node_feature.float())
        reaction_feature = _reaction_one_hot(reaction, self.num_reaction, graph.node_feature.device, graph.num_graph)
        graph_context = torch.cat([output["graph_feature"], reaction_feature], dim=-1)
        node_context = torch.cat(
            [output["node_feature"], graph.node_feature.float(), graph_context[graph.batch]],
            dim=-1,
        )
        edge_context = torch.cat(
            [
                graph.edge_feature.float(),
                node_context[graph.edge_index[0]],
                node_context[graph.edge_index[1]],
            ],
            dim=-1,
        )
        return {
            "node_logits": self.node_mlp(node_context).squeeze(-1),
            "edge_logits": self.edge_mlp(edge_context).squeeze(-1),
            "node_context": node_context,
            "graph_context": graph_context,
        }

    @torch.no_grad()
    def rank_centers(
        self,
        graph: GraphTensor,
        reaction: int | torch.Tensor | None = None,
        topk: int = 5,
    ) -> list[dict]:
        output = self.forward(graph, reaction=reaction)
        candidates = []
        for node_index, score in enumerate(output["node_logits"].tolist()):
            candidates.append(
                {
                    "kind": "atom",
                    "score": float(score),
                    "atom_map_1": int(graph.atom_map[node_index]),
                    "atom_map_2": 0,
                }
            )
        seen_bonds = set()
        for edge_index, score in enumerate(output["edge_logits"].tolist()):
            edge_key = edge_index // 2
            if edge_key in seen_bonds:
                continue
            seen_bonds.add(edge_key)
            src = int(graph.edge_index[0, edge_index])
            dst = int(graph.edge_index[1, edge_index])
            candidates.append(
                {
                    "kind": "bond",
                    "score": float(score),
                    "atom_map_1": int(graph.atom_map[src]),
                    "atom_map_2": int(graph.atom_map[dst]),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:topk]


class SynthonCompletionModel(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        num_relation: int,
        num_reaction: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 3,
        new_atom_symbols: list[str] | None = None,
    ):
        super().__init__()
        self.num_reaction = num_reaction
        self.new_atom_symbols = new_atom_symbols or ATOM_VOCAB
        self.flag_linear = nn.Linear(2, node_input_dim)
        self.encoder = SimpleRGCNEncoder(
            input_dim=node_input_dim,
            hidden_dims=[hidden_dim] * num_layers,
            num_relation=num_relation,
            concat_hidden=True,
        )
        graph_context_dim = self.encoder.output_dim + num_reaction
        self.node_context_dim = self.encoder.output_dim + node_input_dim + graph_context_dim
        self.new_atom_embedding = nn.Embedding(len(self.new_atom_symbols), node_input_dim)
        self.new_atom_linear = nn.Linear(node_input_dim + graph_context_dim, self.node_context_dim)
        self.node_in_mlp = MLP(self.node_context_dim, hidden_dim, 1)
        self.node_out_mlp = MLP(self.node_context_dim * 2, hidden_dim, 1)
        self.bond_mlp = MLP(self.node_context_dim * 2, hidden_dim, len(BOND_TYPE_VOCAB))
        self.stop_mlp = MLP(graph_context_dim, hidden_dim, 1)

    def _build_node_context(
        self,
        graph: GraphTensor,
        reaction_center: tuple[int, ...],
        reaction: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center_mask = reaction_center_mask(graph, reaction_center).to(graph.node_feature.device)
        flags = torch.stack([torch.zeros_like(center_mask, dtype=torch.float32), center_mask.float()], dim=-1)
        node_input = graph.node_feature.float() + self.flag_linear(flags)
        output = self.encoder(graph, node_input)
        reaction_feature = _reaction_one_hot(reaction, self.num_reaction, graph.node_feature.device, graph.num_graph)
        graph_context = torch.cat([output["graph_feature"], reaction_feature], dim=-1)
        node_context = torch.cat(
            [output["node_feature"], graph.node_feature.float(), graph_context[graph.batch]],
            dim=-1,
        )
        return node_context, graph_context

    @torch.no_grad()
    def score_actions(
        self,
        graph: GraphTensor,
        reaction_center: tuple[int, ...],
        reaction: int | torch.Tensor | None = None,
        topk: int = 8,
        num_new_atom_candidate: int = 6,
    ) -> dict:
        if graph.num_graph != 1:
            raise ValueError("Teaching version only supports single-graph action scoring")

        node_context, graph_context = self._build_node_context(graph, reaction_center, reaction=reaction)
        graph_context_single = graph_context[0]
        node_in_logits = self.node_in_mlp(node_context).squeeze(-1)
        node_in_logp = F.log_softmax(node_in_logits, dim=0)
        stop_logit = self.stop_mlp(graph_context).squeeze(-1)[0]
        act_logp = F.logsigmoid(-stop_logit)
        stop_logp = F.logsigmoid(stop_logit)

        num_new_atom_candidate = min(num_new_atom_candidate, len(self.new_atom_symbols))
        new_atom_raw = torch.cat(
            [
                self.new_atom_embedding.weight[:num_new_atom_candidate],
                graph_context_single.unsqueeze(0).repeat(num_new_atom_candidate, 1),
            ],
            dim=-1,
        )
        new_atom_context = self.new_atom_linear(new_atom_raw)
        candidate_context = torch.cat([node_context, new_atom_context], dim=0)

        candidate_desc = []
        for node_idx, atom_map in enumerate(graph.atom_map.tolist()):
            candidate_desc.append(
                {
                    "candidate_kind": "existing_atom",
                    "candidate_index": node_idx,
                    "atom_map": atom_map,
                    "atom_symbol": None,
                }
            )
        for atom_idx, atom_symbol in enumerate(self.new_atom_symbols[:num_new_atom_candidate]):
            candidate_desc.append(
                {
                    "candidate_kind": "new_atom",
                    "candidate_index": graph.num_node + atom_idx,
                    "atom_map": None,
                    "atom_symbol": atom_symbol,
                }
            )

        top_node_in = torch.topk(node_in_logp, k=min(topk, graph.num_node)).indices.tolist()
        actions = []
        for node_in in top_node_in:
            node_in_feature = node_context[node_in].unsqueeze(0).repeat(candidate_context.shape[0], 1)
            pair_feature = torch.cat([node_in_feature, candidate_context], dim=-1)
            node_out_logits = self.node_out_mlp(pair_feature).squeeze(-1)
            if graph.num_node > 1:
                node_out_logits[node_in] = float("-inf")
            node_out_logp = F.log_softmax(node_out_logits, dim=0)
            top_node_out = torch.topk(node_out_logp, k=min(topk, len(candidate_desc))).indices.tolist()
            for node_out in top_node_out:
                bond_logp = F.log_softmax(self.bond_mlp(pair_feature[node_out].unsqueeze(0)).squeeze(0), dim=-1)
                bond_id = int(torch.argmax(bond_logp))
                desc = candidate_desc[node_out]
                actions.append(
                    {
                        "action_type": "edit",
                        "node_in_index": node_in,
                        "node_in_atom_map": int(graph.atom_map[node_in]),
                        "node_out_kind": desc["candidate_kind"],
                        "node_out_index": desc["candidate_index"],
                        "node_out_atom_map": desc["atom_map"],
                        "node_out_symbol": desc["atom_symbol"],
                        "bond_type": BOND_TYPE_NAME[bond_id],
                        "logp": float(node_in_logp[node_in] + node_out_logp[node_out] + bond_logp[bond_id] + act_logp),
                    }
                )

        actions.append(
            {
                "action_type": "stop",
                "node_in_index": None,
                "node_in_atom_map": None,
                "node_out_kind": None,
                "node_out_index": None,
                "node_out_atom_map": None,
                "node_out_symbol": None,
                "bond_type": None,
                "logp": float(stop_logp),
            }
        )
        actions.sort(key=lambda item: item["logp"], reverse=True)
        return {
            "node_in_logits": node_in_logits,
            "stop_logit": stop_logit,
            "top_actions": actions[:topk],
        }


__all__ = [
    "ATOM_VOCAB",
    "BOND_TYPE_NAME",
    "CenterIdentificationModel",
    "GraphTensor",
    "ReactionExample",
    "RELATIVE_TUTORIAL_DIR",
    "SimpleRGCNEncoder",
    "SimpleRGCNLayer",
    "SynthonCompletionModel",
    "TORCHDRUG_SOURCE_MAP",
    "batch_graphs",
    "build_center_identification_dataset",
    "build_synthon_completion_dataset",
    "canonical_smiles_with_mapping",
    "compute_reaction_difference",
    "demo_csv_path",
    "draw_molecule_svg",
    "draw_molecule_image",
    "draw_reaction_pair_image",
    "draw_reaction_pair_svg",
    "export_processed_demo",
    "extract_synthons",
    "feature_dimensions",
    "find_project_root",
    "identify_reaction_center",
    "load_demo_reactions",
    "molecule_to_graph_tensor",
    "oracle_completion_actions",
    "processed_dir",
    "reaction_center_mask",
    "source_mapping_frame",
    "split_reaction_smiles",
    "svg_html",
    "tutorial_dir",
]
