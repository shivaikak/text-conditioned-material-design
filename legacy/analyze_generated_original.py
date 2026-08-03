#!/share/apps/anaconda3/2025.06/bin/python3.13
"""
Compute property statistics for generated SMILES and compare against the
ChEBI training distribution (chebi_material_enriched.jsonl).

Usage
-----
  python analyze_generated.py generated_novel_smiles.txt
  python analyze_generated.py generated_novel_smiles.txt \
      --ref /path/to/chebi_material_enriched.jsonl \
      --output-dir plots/

Outputs
-------
  * Console: validity / uniqueness / novelty rates + per-property stats table
  * plots/<prop>_kde.png : KDE density overlay (generated vs reference)
  * plots/summary_hist.png : all 9 properties in one grid
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless – no display required
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# ── Same 9 descriptors as the notebook's initial data-collection pass ──────────

DESCRIPTOR_NAMES = [
    "MolecularWeight",
    "LogP",
    "TPSA",
    "NumAromaticRings",
    "NumRotatableBonds",
    "FractionCSP3",
    "ApproxSurfaceArea",
    "NumValenceElectrons",
    "HeavyAtomCount",
]

DESCRIPTOR_LABELS = {
    "MolecularWeight":   "Molecular Weight (Da)",
    "LogP":              "LogP (hydrophobicity)",
    "TPSA":              "TPSA (Å²)",
    "NumAromaticRings":  "# Aromatic Rings",
    "NumRotatableBonds": "# Rotatable Bonds",
    "FractionCSP3":      "Fraction sp³ Carbons",
    "ApproxSurfaceArea": "Approx. Surface Area (Å²)",
    "NumValenceElectrons": "# Valence Electrons",
    "HeavyAtomCount":    "Heavy Atom Count",
}


def material_descriptors(mol) -> dict:
    return {
        "MolecularWeight":    Descriptors.MolWt(mol),
        "LogP":               Descriptors.MolLogP(mol),
        "TPSA":               Descriptors.TPSA(mol),
        "NumAromaticRings":   rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumRotatableBonds":  Lipinski.NumRotatableBonds(mol),
        "FractionCSP3":       Descriptors.FractionCSP3(mol),
        "ApproxSurfaceArea":  Descriptors.LabuteASA(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "HeavyAtomCount":     Descriptors.HeavyAtomCount(mol),
    }


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_generated_smiles(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def smiles_to_props(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {"smiles": smiles, **material_descriptors(mol)}


def build_generated_df(smiles_list: list[str]) -> tuple[pd.DataFrame, int]:
    rows, invalid = [], 0
    for s in smiles_list:
        props = smiles_to_props(s)
        if props is None:
            invalid += 1
        else:
            rows.append(props)
    return pd.DataFrame(rows), invalid


def load_reference_df(jsonl_path: Path) -> pd.DataFrame:
    records = [json.loads(ln) for ln in jsonl_path.read_text().splitlines() if ln.strip()]
    df = pd.DataFrame(records)
    # notebook applied HeavyAtomCount < 80 filter before training
    df = df[df["HeavyAtomCount"] < 80].copy()
    return df[["SMILES"] + DESCRIPTOR_NAMES].rename(columns={"SMILES": "smiles"})


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_smiles: list[str],
    valid_smiles: list[str],
    ref_smiles_set: set[str],
) -> dict:
    total = len(all_smiles)
    valid = len(valid_smiles)
    unique = len(set(valid_smiles))
    novel = len(set(valid_smiles) - ref_smiles_set)
    return {
        "total_generated": total,
        "valid":   valid,
        "validity": valid / total if total else 0.0,
        "unique":  unique,
        "uniqueness": unique / valid if valid else 0.0,
        "novel":   novel,
        "novelty":  novel / unique if unique else 0.0,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_kde(ref_df: pd.DataFrame, gen_df: pd.DataFrame, prop: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(ref_df[prop].dropna(), fill=True, alpha=0.5,
                color="steelblue", label="Training (ChEBI)", ax=ax)
    sns.kdeplot(gen_df[prop].dropna(), fill=True, alpha=0.5,
                color="tomato", label="Generated", ax=ax)
    ax.set_xlabel(DESCRIPTOR_LABELS.get(prop, prop))
    ax.set_ylabel("Density")
    ax.set_title(f"{prop} Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary_grid(ref_df: pd.DataFrame, gen_df: pd.DataFrame, out_path: Path):
    ncols = 3
    nrows = -(-len(DESCRIPTOR_NAMES) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, prop in zip(axes, DESCRIPTOR_NAMES):
        sns.kdeplot(ref_df[prop].dropna(), fill=True, alpha=0.5,
                    color="steelblue", label="Training", ax=ax)
        sns.kdeplot(gen_df[prop].dropna(), fill=True, alpha=0.5,
                    color="tomato", label="Generated", ax=ax)
        ax.set_xlabel(DESCRIPTOR_LABELS.get(prop, prop), fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(prop, fontsize=10)
        ax.legend(fontsize=8)

    for ax in axes[len(DESCRIPTOR_NAMES):]:
        ax.set_visible(False)

    fig.suptitle("Generated vs. Training: Property Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Stats table ────────────────────────────────────────────────────────────────

def stats_table(ref_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prop in DESCRIPTOR_NAMES:
        r = ref_df[prop].dropna()
        g = gen_df[prop].dropna()
        rows.append({
            "Property":       prop,
            "Ref mean":       f"{r.mean():.3f}",
            "Ref std":        f"{r.std():.3f}",
            "Ref median":     f"{r.median():.3f}",
            "Gen mean":       f"{g.mean():.3f}" if len(g) else "—",
            "Gen std":        f"{g.std():.3f}"  if len(g) else "—",
            "Gen median":     f"{g.median():.3f}" if len(g) else "—",
        })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

DEFAULT_REF = Path("/scratch/sk10731/text-generated-material-design/chebi_material_enriched.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Property statistics for generated SMILES")
    parser.add_argument("smiles_file", type=Path, help="Text file with one SMILES per line")
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF,
                        help="JSONL reference dataset (default: chebi_material_enriched.jsonl)")
    parser.add_argument("--output-dir", type=Path, default=Path("/scratch/sk10731/text-generated-material-design/claude-supported/plots"),
                        help="Directory for saved plots (default: plots/)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation, print stats only")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading generated SMILES from {args.smiles_file} …")
    all_smiles = load_generated_smiles(args.smiles_file)

    print(f"Computing descriptors for {len(all_smiles)} SMILES …")
    gen_df, n_invalid = build_generated_df(all_smiles)
    valid_smiles = gen_df["smiles"].tolist() if len(gen_df) else []

    print(f"Loading reference dataset from {args.ref} …")
    if not args.ref.exists():
        print(f"  WARNING: reference file not found at {args.ref}; skipping comparison.", file=sys.stderr)
        ref_df = pd.DataFrame(columns=["smiles"] + DESCRIPTOR_NAMES)
        ref_smiles_set: set[str] = set()
    else:
        ref_df = load_reference_df(args.ref)
        ref_smiles_set = set(ref_df["smiles"].tolist())

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = compute_metrics(all_smiles, valid_smiles, ref_smiles_set)

    print("\n── Generation Metrics ───────────────────────────────────────────────")
    print(f"  Total generated : {metrics['total_generated']}")
    print(f"  Valid           : {metrics['valid']}  ({metrics['validity']:.1%})")
    print(f"  Unique (of valid): {metrics['unique']}  ({metrics['uniqueness']:.1%})")
    print(f"  Novel (not in training): {metrics['novel']}  ({metrics['novelty']:.1%})")

    # ── Stats table ────────────────────────────────────────────────────────────
    if len(gen_df) > 0 and len(ref_df) > 0:
        tbl = stats_table(ref_df, gen_df)
        print("\n── Property Statistics ──────────────────────────────────────────────")
        print(tbl.to_string(index=False))

    # ── Plots ──────────────────────────────────────────────────────────────────
    if args.no_plots or len(gen_df) == 0 or len(ref_df) == 0:
        if args.no_plots:
            print("\nPlot generation skipped (--no-plots).")
        else:
            print("\nSkipping plots: generated or reference data is empty.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    print(f"\nSaving plots to {args.output_dir}/ …")
    for prop in DESCRIPTOR_NAMES:
        out = args.output_dir / f"{prop}_kde.png"
        plot_kde(ref_df, gen_df, prop, out)
        print(f"  {out}")

    grid_out = args.output_dir / "summary_grid.png"
    plot_summary_grid(ref_df, gen_df, grid_out)
    print(f"  {grid_out}  (all properties in one figure)")

    print("\nDone.")


if __name__ == "__main__":
    main()
