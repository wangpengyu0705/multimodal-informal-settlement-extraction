#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT = Path.home() / "projects" / "nairobi_v2"

INPUT_CSV = (
    PROJECT
    / "results/final_mapping_v2/seed_42/"
      "21a_full_area_mapping_dataset_final_v2/"
      "full_area_mapping_cellwise_final_v2.csv"
)

OUT_DIR = PROJECT / "results/gsv_availability_audit_v2/seed_42"

OUT_SUMMARY = OUT_DIR / "gsv_availability_summary_by_split_v2.csv"
OUT_CONTINGENCY = OUT_DIR / "gsv_availability_contingency_by_split_v2.csv"
OUT_FIG_PREVALENCE = OUT_DIR / "qc_informal_prevalence_by_gsv_availability_v2.png"
OUT_FIG_COVERAGE = OUT_DIR / "qc_gsv_coverage_rate_by_reference_class_v2.png"
OUT_AUDIT_JSON = OUT_DIR / "gsv_availability_audit_v2.json"
OUT_AUDIT_TXT = OUT_DIR / "gsv_availability_audit_v2.txt"

EXPECTED_VALID = {"train": 67220, "val": 23024, "test": 22356, "all": 112600}
EXPECTED_COVERED = {"train": 15481, "val": 4902, "test": 5744, "all": 26127}


def parse_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    out = (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if out.isna().any():
        bad = series[out.isna()].head(5).tolist()
        raise ValueError(f"Could not parse {name} as boolean: {bad}")
    return out.astype(bool)


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den != 0 else np.nan


def phi_from_2x2(a: int, b: int, c: int, d: int) -> float:
    numerator = (a * d) - (b * c)
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return safe_ratio(numerator, denominator)


def summarize_split(d: pd.DataFrame, split_name: str) -> dict:
    # 2x2 table:
    #                    GSV covered   no GSV
    # informal                a          c
    # non-informal            b          d
    a = int(((d["label"] == 1) & d["has_gsv"]).sum())
    b = int(((d["label"] == 0) & d["has_gsv"]).sum())
    c = int(((d["label"] == 1) & (~d["has_gsv"])).sum())
    dd = int(((d["label"] == 0) & (~d["has_gsv"])).sum())

    n = len(d)
    n_covered = a + b
    n_no_gsv = c + dd
    n_informal = a + c
    n_noninformal = b + dd

    p_informal_given_covered = safe_ratio(a, n_covered)
    p_informal_given_no_gsv = safe_ratio(c, n_no_gsv)
    p_covered_given_informal = safe_ratio(a, n_informal)
    p_covered_given_noninformal = safe_ratio(b, n_noninformal)

    return {
        "split": split_name,
        "n_valid_cells": n,
        "n_gsv_covered": n_covered,
        "n_no_gsv": n_no_gsv,
        "n_informal": n_informal,
        "n_non_informal": n_noninformal,
        "informal_gsv_covered": a,
        "non_informal_gsv_covered": b,
        "informal_no_gsv": c,
        "non_informal_no_gsv": dd,
        "informal_prevalence_gsv_covered": p_informal_given_covered,
        "informal_prevalence_no_gsv": p_informal_given_no_gsv,
        "informal_prevalence_ratio_covered_vs_no_gsv": safe_ratio(
            p_informal_given_covered, p_informal_given_no_gsv
        ),
        "informal_prevalence_difference_percentage_points": (
            p_informal_given_covered - p_informal_given_no_gsv
        ) * 100.0,
        "gsv_coverage_rate_informal": p_covered_given_informal,
        "gsv_coverage_rate_non_informal": p_covered_given_noninformal,
        "gsv_coverage_rate_ratio_informal_vs_non_informal": safe_ratio(
            p_covered_given_informal, p_covered_given_noninformal
        ),
        "gsv_coverage_rate_difference_percentage_points": (
            p_covered_given_informal - p_covered_given_noninformal
        ) * 100.0,
        "odds_ratio_gsv_coverage_informal_vs_non_informal": safe_ratio(
            a * dd, b * c
        ),
        "phi_reference_class_vs_gsv_availability": phi_from_2x2(a, b, c, dd),
    }


def make_prevalence_figure(summary: pd.DataFrame) -> None:
    plot_df = summary.loc[summary["split"].isin(["train", "val", "test", "all"])].copy()
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        x - width / 2,
        plot_df["informal_prevalence_gsv_covered"] * 100.0,
        width,
        label="GSV-covered",
    )
    ax.bar(
        x + width / 2,
        plot_df["informal_prevalence_no_gsv"] * 100.0,
        width,
        label="No GSV",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Train", "Validation", "Test", "All"])
    ax.set_ylabel("Informal-settlement prevalence (%)")
    ax.set_title("QC — Reference-class composition by GSV availability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG_PREVALENCE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_coverage_figure(summary: pd.DataFrame) -> None:
    plot_df = summary.loc[summary["split"].isin(["train", "val", "test", "all"])].copy()
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        x - width / 2,
        plot_df["gsv_coverage_rate_informal"] * 100.0,
        width,
        label="Informal",
    )
    ax.bar(
        x + width / 2,
        plot_df["gsv_coverage_rate_non_informal"] * 100.0,
        width,
        label="Non-informal",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Train", "Validation", "Test", "All"])
    ax.set_ylabel("GSV coverage rate (%)")
    ax.set_title("QC — GSV coverage rate by reference class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG_COVERAGE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    protected = [OUT_SUMMARY, OUT_CONTINGENCY, OUT_AUDIT_JSON]
    existing = [p for p in protected if p.exists()]
    if existing:
        raise FileExistsError(
            "Protected availability-audit outputs already exist; refusing overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    required = {"split", "label", "has_gsv", "global_cell_row", "global_cell_col"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["has_gsv"] = parse_bool(df["has_gsv"], "has_gsv")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(np.int8)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels: {sorted(df['label'].unique())}")

    if len(df) != EXPECTED_VALID["all"]:
        raise ValueError(f"Rows {len(df)} != expected {EXPECTED_VALID['all']}")

    if df.duplicated(["global_cell_row", "global_cell_col"]).any():
        raise ValueError("Duplicate global cells")

    for split in ["train", "val", "test"]:
        n = int((df["split"] == split).sum())
        if n != EXPECTED_VALID[split]:
            raise ValueError(f"{split}: valid rows {n} != {EXPECTED_VALID[split]}")
        covered = int(df.loc[df["split"] == split, "has_gsv"].sum())
        if covered != EXPECTED_COVERED[split]:
            raise ValueError(
                f"{split}: covered rows {covered} != {EXPECTED_COVERED[split]}"
            )

    if int(df["has_gsv"].sum()) != EXPECTED_COVERED["all"]:
        raise ValueError("All-area GSV-covered count mismatch")

    rows = [
        summarize_split(df.loc[df["split"] == split].copy(), split)
        for split in ["train", "val", "test"]
    ]
    rows.append(summarize_split(df.copy(), "all"))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    contingency_rows = []
    for _, r in summary.iterrows():
        contingency_rows.extend(
            [
                {
                    "split": r["split"],
                    "reference_class": "informal",
                    "gsv_covered": int(r["informal_gsv_covered"]),
                    "no_gsv": int(r["informal_no_gsv"]),
                },
                {
                    "split": r["split"],
                    "reference_class": "non_informal",
                    "gsv_covered": int(r["non_informal_gsv_covered"]),
                    "no_gsv": int(r["non_informal_no_gsv"]),
                },
            ]
        )
    pd.DataFrame(contingency_rows).to_csv(OUT_CONTINGENCY, index=False)

    make_prevalence_figure(summary)
    make_coverage_figure(summary)

    split_rows = summary.loc[summary["split"].isin(["train", "val", "test"])]
    split_direction_consistent = bool(
        (
            split_rows["informal_prevalence_gsv_covered"].to_numpy()
            > split_rows["informal_prevalence_no_gsv"].to_numpy()
        ).all()
    )

    test_row = summary.loc[summary["split"] == "test"].iloc[0]
    all_row = summary.loc[summary["split"] == "all"].iloc[0]

    audit = {
        "status": "PASSED",
        "purpose": (
            "Descriptive audit of whether GSV availability is associated with "
            "reference-class composition. This is not a model-performance experiment "
            "and does not identify the causal mechanism of GSV coverage."
        ),
        "new_model_training": False,
        "new_model_inference": False,
        "threshold_tuning": False,
        "causal_claim_about_coverage": False,
        "naive_independence_p_value_reported": False,
        "reason_no_naive_p_value": (
            "Cell-level spatial dependence makes an ordinary independence-test "
            "p-value difficult to interpret. The audit emphasizes effect sizes "
            "and consistency across Train/Validation/Test."
        ),
        "split_direction_consistent_train_val_test": split_direction_consistent,
        "primary_heldout_test": {
            "n_gsv_covered": int(test_row["n_gsv_covered"]),
            "n_no_gsv": int(test_row["n_no_gsv"]),
            "informal_prevalence_gsv_covered": float(
                test_row["informal_prevalence_gsv_covered"]
            ),
            "informal_prevalence_no_gsv": float(
                test_row["informal_prevalence_no_gsv"]
            ),
            "prevalence_ratio_covered_vs_no_gsv": float(
                test_row["informal_prevalence_ratio_covered_vs_no_gsv"]
            ),
            "gsv_coverage_rate_informal": float(
                test_row["gsv_coverage_rate_informal"]
            ),
            "gsv_coverage_rate_non_informal": float(
                test_row["gsv_coverage_rate_non_informal"]
            ),
            "coverage_rate_ratio_informal_vs_non_informal": float(
                test_row["gsv_coverage_rate_ratio_informal_vs_non_informal"]
            ),
            "odds_ratio_gsv_coverage_informal_vs_non_informal": float(
                test_row["odds_ratio_gsv_coverage_informal_vs_non_informal"]
            ),
            "phi_reference_class_vs_gsv_availability": float(
                test_row["phi_reference_class_vs_gsv_availability"]
            ),
        },
        "full_dataset_descriptive": {
            "n_valid_cells": int(all_row["n_valid_cells"]),
            "n_gsv_covered": int(all_row["n_gsv_covered"]),
            "n_no_gsv": int(all_row["n_no_gsv"]),
            "informal_prevalence_gsv_covered": float(
                all_row["informal_prevalence_gsv_covered"]
            ),
            "informal_prevalence_no_gsv": float(
                all_row["informal_prevalence_no_gsv"]
            ),
            "prevalence_ratio_covered_vs_no_gsv": float(
                all_row["informal_prevalence_ratio_covered_vs_no_gsv"]
            ),
        },
        "interpretation_boundary": (
            "The audit can support that GSV-covered and no-GSV cells have different "
            "reference-class composition, and therefore that GSV availability acts "
            "as a sample-selection constraint. It does NOT show why Google Street "
            "View coverage has this pattern, nor that informal-settlement status "
            "causes GSV availability."
        ),
        "outputs": {
            "summary_csv": str(OUT_SUMMARY),
            "contingency_csv": str(OUT_CONTINGENCY),
            "prevalence_qc_png": str(OUT_FIG_PREVALENCE),
            "coverage_qc_png": str(OUT_FIG_COVERAGE),
        },
    }

    OUT_AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    OUT_AUDIT_TXT.write_text(
        "\n".join(
            [
                "=" * 92,
                "Nairobi v2 — GSV Availability / Missingness Audit",
                "=" * 92,
                "STATUS: PASSED",
                "",
                "Training: NO",
                "New inference: NO",
                "Threshold tuning: NO",
                "Causal coverage claim: NO",
                "Naive independence p-value: NO",
                "",
                "Summary:",
                summary.to_string(index=False),
                "",
                f"Train/Val/Test direction consistent: {split_direction_consistent}",
                "",
                "Interpretation boundary:",
                "  Association / sample-selection evidence only.",
                "  This audit does not identify the mechanism causing GSV coverage.",
                "",
                f"Summary CSV: {OUT_SUMMARY}",
                f"Contingency CSV: {OUT_CONTINGENCY}",
                f"Prevalence QC: {OUT_FIG_PREVALENCE}",
                f"Coverage QC: {OUT_FIG_COVERAGE}",
            ]
        ),
        encoding="utf-8",
    )

    print("=" * 92)
    print("Nairobi v2 — GSV Availability / Missingness Audit")
    print("=" * 92)
    print("STATUS: PASSED")
    print()
    print(summary.to_string(index=False))
    print()
    print("Train/Val/Test direction consistent:", split_direction_consistent)
    print()
    print("Primary held-out Test:")
    print(
        "  Informal prevalence | GSV-covered:",
        f"{100.0 * test_row['informal_prevalence_gsv_covered']:.3f}%",
    )
    print(
        "  Informal prevalence | no GSV:",
        f"{100.0 * test_row['informal_prevalence_no_gsv']:.3f}%",
    )
    print(
        "  Prevalence ratio:",
        f"{test_row['informal_prevalence_ratio_covered_vs_no_gsv']:.3f}",
    )
    print(
        "  GSV coverage rate | informal:",
        f"{100.0 * test_row['gsv_coverage_rate_informal']:.3f}%",
    )
    print(
        "  GSV coverage rate | non-informal:",
        f"{100.0 * test_row['gsv_coverage_rate_non_informal']:.3f}%",
    )
    print(
        "  Coverage-rate ratio:",
        f"{test_row['gsv_coverage_rate_ratio_informal_vs_non_informal']:.3f}",
    )
    print(
        "  Coverage odds ratio:",
        f"{test_row['odds_ratio_gsv_coverage_informal_vs_non_informal']:.3f}",
    )
    print(
        "  Phi:",
        f"{test_row['phi_reference_class_vs_gsv_availability']:.3f}",
    )
    print()
    print("Do NOT infer the causal mechanism of GSV coverage from this audit.")
    print()
    print("Outputs:")
    print(" ", OUT_SUMMARY)
    print(" ", OUT_CONTINGENCY)
    print(" ", OUT_FIG_PREVALENCE)
    print(" ", OUT_FIG_COVERAGE)
    print(" ", OUT_AUDIT_JSON)


if __name__ == "__main__":
    main()
