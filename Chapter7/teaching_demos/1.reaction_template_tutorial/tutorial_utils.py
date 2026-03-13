from __future__ import annotations

import html
import importlib
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from IPython.display import HTML
from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions


def format_project_path(path: str | Path, project_root: Path) -> str:
    path_obj = Path(path).resolve()
    project_root = project_root.resolve()
    try:
        relative = path_obj.relative_to(project_root)
        return "project/" if not relative.parts else f"project/{relative.as_posix()}"
    except ValueError:
        return path_obj.name or str(path_obj)


def find_project_root(start: Path | None = None) -> Path:
    base = (start or Path.cwd()).resolve()
    if base.is_file():
        base = base.parent

    for candidate in [base, *base.parents]:
        if (
            (candidate / "source_repos" / "rdchiral").exists()
            and (candidate / "source_repos" / "rxnmapper").exists()
        ):
            return candidate

    raise FileNotFoundError("无法定位同时包含 source_repos/rdchiral 和 source_repos/rxnmapper 的项目根目录。")


def ensure_source_repos_on_path(project_root: Path) -> dict[str, Path]:
    repo_paths = {
        "rdchiral": project_root / "source_repos" / "rdchiral",
        "rxnmapper": project_root / "source_repos" / "rxnmapper",
    }

    for repo_path in repo_paths.values():
        repo_text = str(repo_path)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)

    return repo_paths


def environment_report(project_root: Path) -> pd.DataFrame:
    rows = [
        {"组件": "python", "版本": sys.version.split()[0], "位置": format_project_path(sys.executable, project_root)},
        {"组件": "project_root", "版本": "-", "位置": "project/"},
        {
            "组件": "tutorial_dir",
            "版本": "-",
            "位置": format_project_path(project_root / "teaching_demos" / "reaction_template_tutorial", project_root),
        },
        {
            "组件": "rdchiral_repo",
            "版本": "-",
            "位置": format_project_path(project_root / "source_repos" / "rdchiral", project_root),
        },
        {
            "组件": "rxnmapper_repo",
            "版本": "-",
            "位置": format_project_path(project_root / "source_repos" / "rxnmapper", project_root),
        },
    ]

    for module_name in ["pandas", "rdkit", "torch", "transformers", "rxnmapper", "rdchiral"]:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "n/a")
        location = format_project_path(getattr(module, "__file__", ""), project_root)
        rows.append({"组件": module_name, "版本": version, "位置": location})

    return pd.DataFrame(rows)


def split_smiles_side(side: str) -> list[str]:
    return [fragment for fragment in side.split(".") if fragment]


def _prepare_mol(smiles: str, annotate_atom_maps: bool = False) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")

    mol = Chem.Mol(mol)
    if annotate_atom_maps:
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map:
                atom.SetProp("atomNote", str(atom_map))
    return mol


def draw_molecule_grid_svg(
    smiles_list: Sequence[str],
    legends: Sequence[str] | None = None,
    annotate_atom_maps: bool = False,
    mols_per_row: int = 3,
    sub_img_size: tuple[int, int] = (280, 220),
) -> str:
    mols = [_prepare_mol(smiles, annotate_atom_maps=annotate_atom_maps) for smiles in smiles_list]
    if legends is None:
        legends = [""] * len(mols)
    return _svg_fragment(
        str(
            Draw.MolsToGridImage(
                mols,
                legends=list(legends),
                molsPerRow=mols_per_row,
                subImgSize=sub_img_size,
                useSVG=True,
            )
        )
    )


def draw_reaction_svg(
    reaction_text: str,
    *,
    use_smiles: bool,
    sub_img_size: tuple[int, int] = (260, 180),
) -> str:
    rxn = rdChemReactions.ReactionFromSmarts(reaction_text, useSmiles=use_smiles)
    if rxn is None:
        raise ValueError(f"无法解析反应表示: {reaction_text}")
    return _svg_fragment(str(Draw.ReactionToImage(rxn, subImgSize=sub_img_size, useSVG=True)))


def draw_retro_route_svg(
    target_product_smiles: str,
    precursor_smiles: str,
    sub_img_size: tuple[int, int] = (240, 170),
) -> str:
    return draw_reaction_svg(f"{target_product_smiles}>>{precursor_smiles}", use_smiles=True, sub_img_size=sub_img_size)


def run_template_workflow(
    unmapped_rxn_smiles: str,
    target_product_smiles: str,
    *,
    reaction_id: str = "teaching_demo",
    mapper=None,
) -> dict:
    from rxnmapper import RXNMapper
    from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
    from rdchiral.template_extractor import extract_from_reaction

    mapper = mapper or RXNMapper()
    mapping_result = mapper.get_attention_guided_atom_maps([unmapped_rxn_smiles])[0]
    mapped_reaction_smiles = mapping_result["mapped_rxn"]
    mapped_reactants, mapped_products = mapped_reaction_smiles.split(">>")

    reaction_record = {
        "_id": reaction_id,
        "reactants": mapped_reactants,
        "products": mapped_products,
    }
    template = extract_from_reaction(reaction_record)
    reaction_template_smarts = template["reaction_smarts"]

    rxn = rdchiralReaction(reaction_template_smarts)
    outcomes = sorted(rdchiralRun(rxn, rdchiralReactants(target_product_smiles)))

    return {
        "input_reaction_smiles": unmapped_rxn_smiles,
        "mapped_reaction_smiles": mapped_reaction_smiles,
        "mapping_confidence": mapping_result["confidence"],
        "reaction_record": reaction_record,
        "template": template,
        "reaction_template_smarts": reaction_template_smarts,
        "target_product_smiles": target_product_smiles,
        "outcomes": outcomes,
    }


def evaluate_template_targets(
    reaction_template_smarts: str,
    labelled_targets: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun

    rxn = rdchiralReaction(reaction_template_smarts)
    rows = []
    for label, target_smiles in labelled_targets:
        outcomes = sorted(rdchiralRun(rxn, rdchiralReactants(target_smiles)))
        rows.append(
            {
                "label": label,
                "target_product_smiles": target_smiles,
                "match_count": len(outcomes),
                "first_outcome": outcomes[0] if outcomes else "",
                "all_outcomes": outcomes,
            }
        )
    return pd.DataFrame(rows)


def route_gallery_html(records: Iterable[dict]) -> HTML:
    cards = []
    for record in records:
        label = html.escape(str(record.get("label", "案例")))
        target_smiles = str(record.get("target_product_smiles", ""))
        first_outcome = str(record.get("first_outcome", ""))
        match_count = int(record.get("match_count", 0))

        if first_outcome:
            route_svg = _svg_fragment(draw_retro_route_svg(target_smiles, first_outcome))
            outcome_line = html.escape(first_outcome)
        else:
            route_svg = "<div style='padding: 20px 0; color: #666;'>当前模板没有匹配到候选前体。</div>"
            outcome_line = "无匹配"

        count_line = f"{match_count} 条候选" if match_count else "0 条候选"
        target_line = html.escape(target_smiles)
        cards.append(
            f"""
            <div style="border: 1px solid #d0d7de; border-radius: 14px; padding: 14px; background: #ffffff;">
              <div style="font-size: 16px; font-weight: 700; margin-bottom: 6px;">{label}</div>
              <div style="font-size: 12px; color: #475467; margin-bottom: 4px; word-break: break-all;">
                Target: <code>{target_line}</code>
              </div>
              <div style="font-size: 12px; color: #475467; margin-bottom: 4px;">匹配结果: {count_line}</div>
              <div style="font-size: 12px; color: #475467; margin-bottom: 10px; word-break: break-all;">
                First outcome: <code>{outcome_line}</code>
              </div>
              <div style="overflow-x: auto;">{route_svg}</div>
            </div>
            """
        )

    html_block = (
        "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); "
        "gap: 12px; margin: 8px 0 16px;'>"
        + "".join(cards)
        + "</div>"
    )
    return HTML(html_block)


def _svg_fragment(svg: str) -> str:
    if svg.startswith("<?xml"):
        return svg.split("?>", maxsplit=1)[1]
    return svg
