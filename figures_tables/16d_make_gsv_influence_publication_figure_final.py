#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

PROJECT = Path.home() / "projects" / "nairobi_v2"

INPUT_CSV = (
    PROJECT
    / "results/xai_modality_v2/seed_42/"
      "xai_modality_delta_cellwise_v2.csv"
)

OUT_DIR = (
    PROJECT
    / "results/xai_modality_v2/seed_42/"
      "16d_publication_figures_final"
)

FIG_PNG = OUT_DIR / "figure_gsv_influence_by_predictive_consequence_final.png"
FIG_PDF = OUT_DIR / "figure_gsv_influence_by_predictive_consequence_final.pdf"
FIG_TIF = OUT_DIR / "figure_gsv_influence_by_predictive_consequence_final.tif"
SUMMARY_CSV = OUT_DIR / "figure_gsv_influence_by_predictive_consequence_summary_final.csv"
AUDIT_JSON = OUT_DIR / "figure_gsv_influence_by_predictive_consequence_audit_final.json"

EXPECTED_N = 5744
EXPECTED_GROUP_COUNTS = {
    "Stable correct": 4929,
    "Stable incorrect": 383,
    "Helpful": 238,
    "Harmful": 194,
}

GROUP_ORDER = [
    "Stable correct",
    "Stable incorrect",
    "Helpful",
    "Harmful",
]

# Muted, color-blind-friendly publication palette.
GROUP_COLORS = {
    "Stable correct": "#4C78A8",
    "Stable incorrect": "#9E9E9E",
    "Helpful": "#59A14F",
    "Harmful": "#E15759",
}

# Journal-oriented canvas.
FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 105.0

PNG_DPI = 600
TIFF_DPI = 600

FONT_FAMILY = [
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "DejaVu Sans",
]


# =============================================================================
# Helpers
# =============================================================================

def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 0.7,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def choose_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find {label}. Tried: {candidates}")


def assign_group(label: int, s2_pred: int, fusion_pred: int) -> str:
    s2_correct = int(s2_pred) == int(label)
    fusion_correct = int(fusion_pred) == int(label)
    changed = int(s2_pred) != int(fusion_pred)

    if (not changed) and s2_correct and fusion_correct:
        return "Stable correct"

    if (not changed) and (not s2_correct) and (not fusion_correct):
        return "Stable incorrect"

    if changed and (not s2_correct) and fusion_correct:
        return "Helpful"

    if changed and s2_correct and (not fusion_correct):
        return "Harmful"

    raise ValueError(
        "Unexpected binary outcome combination: "
        f"label={label}, s2={s2_pred}, fusion={fusion_pred}"
    )


def load_data():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    if len(df) != EXPECTED_N:
        raise ValueError(f"Rows {len(df)} != expected {EXPECTED_N}")

    label_col = choose_column(
        df,
        ["label", "reference_label"],
        "reference label",
    )
    s2_col = choose_column(
        df,
        ["s2_prediction"],
        "S2 prediction",
    )
    fusion_col = choose_column(
        df,
        ["fusion_prediction", "fusion_mean_prediction", "final_prediction"],
        "Fusion prediction",
    )
    abs_col = choose_column(
        df,
        ["abs_delta_gsv_logit", "abs_delta_gsv", "abs_delta"],
        "absolute ΔGSV",
    )
    signed_col = choose_column(
        df,
        ["delta_gsv_logit", "delta_gsv"],
        "signed ΔGSV",
    )

    for col in [label_col, s2_col, fusion_col]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    for col in [abs_col, signed_col]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

    if not np.isfinite(df[abs_col].to_numpy()).all():
        raise ValueError("Absolute ΔGSV contains non-finite values")

    if not np.isfinite(df[signed_col].to_numpy()).all():
        raise ValueError("Signed ΔGSV contains non-finite values")

    df["predictive_consequence_group"] = [
        assign_group(label, s2_pred, fusion_pred)
        for label, s2_pred, fusion_pred in zip(
            df[label_col].to_numpy(),
            df[s2_col].to_numpy(),
            df[fusion_col].to_numpy(),
        )
    ]

    counts = (
        df["predictive_consequence_group"]
        .value_counts()
        .reindex(GROUP_ORDER)
        .fillna(0)
        .astype(int)
        .to_dict()
    )

    if counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            f"Group counts mismatch: {counts} != {EXPECTED_GROUP_COUNTS}"
        )

    return df, label_col, s2_col, fusion_col, abs_col, signed_col


def build_summary(
    df: pd.DataFrame,
    abs_col: str,
    signed_col: str,
) -> pd.DataFrame:
    rows = []

    for group in GROUP_ORDER:
        sub = df.loc[
            df["predictive_consequence_group"] == group
        ].copy()

        abs_vals = sub[abs_col].to_numpy(dtype=float)
        signed_vals = sub[signed_col].to_numpy(dtype=float)

        rows.append({
            "group": group,
            "n": int(len(sub)),
            "abs_mean": float(np.mean(abs_vals)),
            "abs_q25": float(np.quantile(abs_vals, 0.25)),
            "abs_median": float(np.median(abs_vals)),
            "abs_q75": float(np.quantile(abs_vals, 0.75)),
            "abs_p90": float(np.quantile(abs_vals, 0.90)),
            "abs_p95": float(np.quantile(abs_vals, 0.95)),
            "abs_max": float(np.max(abs_vals)),
            "signed_mean": float(np.mean(signed_vals)),
            "signed_median": float(np.median(signed_vals)),
            "positive_pct": float(100.0 * np.mean(signed_vals > 0)),
            "negative_pct": float(100.0 * np.mean(signed_vals < 0)),
        })

    return pd.DataFrame(rows)


def make_figure(
    df: pd.DataFrame,
    abs_col: str,
    summary: pd.DataFrame,
) -> None:
    setup_matplotlib()

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH_MM / 25.4,
            FIG_HEIGHT_MM / 25.4,
        ),
        constrained_layout=False,
    )

    positions = np.arange(
        1,
        len(GROUP_ORDER) + 1,
    )

    data = [
        df.loc[
            df["predictive_consequence_group"] == group,
            abs_col,
        ].to_numpy(dtype=float)
        for group in GROUP_ORDER
    ]

    # -----------------------------------------------------------------
    # Violin distributions
    # -----------------------------------------------------------------
    violin = ax.violinplot(
        data,
        positions=positions,
        widths=0.80,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
    )

    for body, group in zip(
        violin["bodies"],
        GROUP_ORDER,
    ):
        body.set_facecolor(
            GROUP_COLORS[group]
        )
        body.set_edgecolor(
            "#555555"
        )
        body.set_alpha(
            0.28
        )
        body.set_linewidth(
            0.65
        )

    # -----------------------------------------------------------------
    # Embedded boxplots
    # -----------------------------------------------------------------
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={
            "color": "black",
            "linewidth": 1.15,
        },
        boxprops={
            "edgecolor": "black",
            "linewidth": 0.75,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 0.70,
        },
        capprops={
            "color": "black",
            "linewidth": 0.70,
        },
    )

    for patch, group in zip(
        box["boxes"],
        GROUP_ORDER,
    ):
        patch.set_facecolor(
            GROUP_COLORS[group]
        )
        patch.set_alpha(
            0.86
        )

    # -----------------------------------------------------------------
    # Median labels
    # -----------------------------------------------------------------
    summary_idx = summary.set_index("group")

    for x, group in zip(
        positions,
        GROUP_ORDER,
    ):
        median = float(
            summary_idx.loc[
                group,
                "abs_median",
            ]
        )

        q3 = float(
            summary_idx.loc[
                group,
                "abs_q75",
            ]
        )

        ax.text(
            x,
            q3 + 0.12,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color="black",
        )

    # -----------------------------------------------------------------
    # Axis styling
    # -----------------------------------------------------------------
    ax.set_ylabel(
        "Absolute GSV influence, |ΔGSV| (logit difference)"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        [
            f"{group}\n(n={EXPECTED_GROUP_COUNTS[group]:,})"
            for group in GROUP_ORDER
        ],
        linespacing=1.05,
    )

    ymax = max(
        float(np.max(vals))
        for vals in data
    )

    ax.set_ylim(
        0.0,
        max(8.8, ymax * 1.045),
    )

    # Light horizontal guides only.
    ax.yaxis.grid(
        True,
        color="#E5E5E5",
        linewidth=0.55,
        linestyle="-",
    )
    ax.xaxis.grid(
        False
    )
    ax.set_axisbelow(
        True
    )

    # Journal-style open frame.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.tick_params(
        axis="x",
        length=0,
        pad=6,
    )
    ax.tick_params(
        axis="y",
        length=3.2,
        width=0.6,
    )

    # No panel label: this is a standalone figure.
    # No title: descriptive title belongs in the manuscript caption.

    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.97,
        bottom=0.19,
    )

    fig.savefig(
        FIG_PDF,
        facecolor="white",
    )
    fig.savefig(
        FIG_PNG,
        dpi=PNG_DPI,
        facecolor="white",
    )
    fig.savefig(
        FIG_TIF,
        dpi=TIFF_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    plt.close(fig)


def main():
    print("=" * 96)
    print("Nairobi v2 — Step16 Final Publication Figure")
    print("=" * 96)
    print("New model training: NO")
    print("New model inference: NO")
    print("ΔGSV recomputation: NO")
    print("Purpose: publication-quality visualization of frozen Step16 outputs")
    print()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    protected = [
        FIG_PNG,
        FIG_PDF,
        FIG_TIF,
        SUMMARY_CSV,
        AUDIT_JSON,
    ]

    existing = [
        p
        for p in protected
        if p.exists()
    ]

    if existing:
        raise FileExistsError(
            "Protected final Step16 outputs already exist; refusing overwrite:\n"
            + "\n".join(
                str(p)
                for p in existing
            )
        )

    (
        df,
        label_col,
        s2_col,
        fusion_col,
        abs_col,
        signed_col,
    ) = load_data()

    summary = build_summary(
        df,
        abs_col,
        signed_col,
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    make_figure(
        df,
        abs_col,
        summary,
    )

    audit = {
        "status": "PASSED",
        "step": "16d_final",
        "purpose": (
            "Publication-quality violin + embedded boxplot of frozen "
            "held-out Test |ΔGSV| values by predictive consequence."
        ),
        "scientific_recomputation": False,
        "new_model_training": False,
        "new_model_inference": False,
        "delta_gsv_recomputed": False,
        "source_csv": str(INPUT_CSV),
        "n_cells": int(len(df)),
        "reference_label_column": label_col,
        "s2_prediction_column": s2_col,
        "fusion_prediction_column": fusion_col,
        "abs_delta_column": abs_col,
        "signed_delta_column": signed_col,
        "group_order": GROUP_ORDER,
        "group_counts": EXPECTED_GROUP_COUNTS,
        "figure_design": {
            "figure_width_mm": FIG_WIDTH_MM,
            "figure_height_mm": FIG_HEIGHT_MM,
            "palette": GROUP_COLORS,
            "violin_alpha": 0.28,
            "boxplot_fliers": False,
            "median_labels": True,
            "standalone_panel_label": False,
            "standalone_title": False,
            "ylabel": (
                "Absolute GSV influence, |ΔGSV| (logit difference)"
            ),
            "png_dpi": PNG_DPI,
            "tiff_dpi": TIFF_DPI,
            "pdf_fonttype": 42,
        },
        "interpretation_boundary": (
            "The figure is descriptive. Absolute ΔGSV measures the magnitude "
            "of the GSV modality effect on the Fusion logit relative to the "
            "GSV-neutralized representation. A larger |ΔGSV| does not by itself "
            "imply a beneficial prediction change, and a hard-class transition "
            "depends on both influence magnitude and proximity to the decision threshold."
        ),
        "outputs": {
            "figure_png": str(FIG_PNG),
            "figure_pdf": str(FIG_PDF),
            "figure_tif": str(FIG_TIF),
            "summary_csv": str(SUMMARY_CSV),
            "audit_json": str(AUDIT_JSON),
        },
    }

    AUDIT_JSON.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("STATUS: PASSED")
    print()
    print(summary.to_string(index=False))
    print()
    print("Outputs:")
    print(" ", FIG_PNG)
    print(" ", FIG_PDF)
    print(" ", FIG_TIF)
    print(" ", SUMMARY_CSV)
    print(" ", AUDIT_JSON)


if __name__ == "__main__":
    main()
