from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from common import load_yaml, utc_now, write_json

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_FUNCTIONS = {
    "MolecularWeight": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "FractionCSP3": Descriptors.FractionCSP3,
    "ApproxSurfaceArea": Descriptors.LabuteASA,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}


def load_smiles(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def descriptor_row(smiles: str, descriptors: list[str]) -> dict[str, Any] | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    row: dict[str, Any] = {"smiles": smiles}
    for descriptor in descriptors:
        row[descriptor] = float(DESCRIPTOR_FUNCTIONS[descriptor](molecule))
    return row


def generated_frame(smiles: list[str], descriptors: list[str]) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    for item in smiles:
        row = descriptor_row(item, descriptors)
        if row is None:
            invalid += 1
        else:
            rows.append(row)
    return pd.DataFrame(rows), invalid


def reference_frame(path: Path, descriptors: list[str], max_heavy_atoms: int) -> pd.DataFrame:
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    frame = pd.DataFrame(records)
    frame = frame[frame["HeavyAtomCount"] < max_heavy_atoms].copy()
    required = ["SMILES", *descriptors]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Reference dataset is missing columns: {missing}")
    return frame[required].rename(columns={"SMILES": "smiles"})


def rank_biserial_from_u(u_statistic: float, n_ref: int, n_gen: int) -> float:
    return 1.0 - (2.0 * u_statistic) / (n_ref * n_gen)


def bootstrap_mean_difference(
    reference: np.ndarray,
    generated: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for index in range(iterations):
        ref_sample = rng.choice(reference, size=len(reference), replace=True)
        gen_sample = rng.choice(generated, size=len(generated), replace=True)
        differences[index] = gen_sample.mean() - ref_sample.mean()
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return float(differences.mean()), float(lower), float(upper)


def statistical_tests(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
    descriptors: list[str],
    alpha: float,
    correction: str,
    bootstrap_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for offset, descriptor in enumerate(descriptors):
        ref_values = reference[descriptor].dropna().to_numpy(dtype=float)
        gen_values = generated[descriptor].dropna().to_numpy(dtype=float)
        if len(ref_values) == 0 or len(gen_values) == 0:
            raise ValueError(f"No values available for {descriptor}")
        test = mannwhitneyu(ref_values, gen_values, alternative="two-sided", method="auto")
        effect = rank_biserial_from_u(float(test.statistic), len(ref_values), len(gen_values))
        boot_mean, boot_low, boot_high = bootstrap_mean_difference(
            ref_values,
            gen_values,
            iterations=bootstrap_iterations,
            seed=seed + offset,
        )
        raw_p_values.append(float(test.pvalue))
        rows.append(
            {
                "descriptor": descriptor,
                "reference_n": len(ref_values),
                "generated_n": len(gen_values),
                "reference_mean": float(ref_values.mean()),
                "generated_mean": float(gen_values.mean()),
                "reference_median": float(np.median(ref_values)),
                "generated_median": float(np.median(gen_values)),
                "u_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "rank_biserial_effect": effect,
                "bootstrap_mean_difference": boot_mean,
                "bootstrap_ci_95_low": boot_low,
                "bootstrap_ci_95_high": boot_high,
            }
        )
    rejected, adjusted, _, _ = multipletests(raw_p_values, alpha=alpha, method=correction)
    for row, is_rejected, adjusted_p in zip(rows, rejected, adjusted):
        row["adjusted_p_value"] = float(adjusted_p)
        row["significantly_different"] = bool(is_rejected)
    return rows


def standardized_mean_distance(
    reference: pd.DataFrame, generated: pd.DataFrame, descriptors: list[str]
) -> float:
    distances: list[float] = []
    for descriptor in descriptors:
        ref = reference[descriptor].dropna().to_numpy(dtype=float)
        gen = generated[descriptor].dropna().to_numpy(dtype=float)
        pooled_scale = max(float(np.std(ref, ddof=1)), 1e-12)
        distances.append(abs(float(np.mean(gen) - np.mean(ref))) / pooled_scale)
    return float(np.mean(distances))


def plot_distributions(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
    descriptors: list[str],
    output_dir: Path,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for descriptor in descriptors:
        figure, axis = plt.subplots(figsize=(6, 4))
        sns.kdeplot(reference[descriptor].dropna(), fill=True, alpha=0.4, label="Reference", ax=axis)
        sns.kdeplot(generated[descriptor].dropna(), fill=True, alpha=0.4, label="Generated", ax=axis)
        axis.set_title(f"{descriptor}: generated vs reference")
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / f"{descriptor}_kde.png", dpi=160)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    evaluation = config["evaluation"]
    descriptors = list(evaluation["descriptors"])
    seed = int(config["experiment"]["seed"])
    reference_path = Path(config["paths"]["reference"])
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference data not found: {reference_path}")

    all_smiles = load_smiles(args.generated)
    generated, invalid_count = generated_frame(all_smiles, descriptors)
    if generated.empty:
        raise RuntimeError("No valid generated molecules were available for evaluation")
    reference = reference_frame(
        reference_path,
        descriptors,
        int(config["filters"]["max_heavy_atoms"]),
    )
    reference_smiles = set(reference["smiles"])
    valid_smiles = generated["smiles"].tolist()
    unique_smiles = set(valid_smiles)
    novel_smiles = unique_smiles - reference_smiles

    tests = statistical_tests(
        reference,
        generated,
        descriptors,
        alpha=float(evaluation["alpha"]),
        correction=str(evaluation["multiple_testing"]),
        bootstrap_iterations=int(evaluation["bootstrap_iterations"]),
        seed=seed,
    )
    significant_count = sum(row["significantly_different"] for row in tests)
    mean_distance = standardized_mean_distance(reference, generated, descriptors)
    conformity_proxy = 1.0 / (1.0 + mean_distance)

    output = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "evaluated_at": utc_now(),
        "generation": {
            "total": len(all_smiles),
            "invalid_count": invalid_count,
            "valid_count": len(valid_smiles),
            "validity": len(valid_smiles) / len(all_smiles) if all_smiles else 0.0,
            "unique_count": len(unique_smiles),
            "uniqueness": len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0,
            "novel_count": len(novel_smiles),
            "novelty": len(novel_smiles) / len(unique_smiles) if unique_smiles else 0.0,
        },
        "distribution": {
            "standardized_mean_distance": mean_distance,
            "conformity_proxy": conformity_proxy,
            "descriptors_significantly_different": significant_count,
            "descriptor_count": len(descriptors),
        },
        "mann_whitney": tests,
        "interpretation_warning": (
            "Failure to reject a Mann-Whitney U null hypothesis does not prove equal distributions. "
            "Interpret adjusted p-values with effect sizes, confidence intervals, plots, and multivariate metrics."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "evaluation.json", output)
    pd.DataFrame(tests).to_csv(args.output_dir / "descriptor_tests.csv", index=False)
    generated.to_csv(args.output_dir / "generated_descriptors.csv", index=False)
    if not args.no_plots:
        plot_distributions(reference, generated, descriptors, args.output_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
