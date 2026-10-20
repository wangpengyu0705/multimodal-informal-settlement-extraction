#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer


PROJECT = Path.home() / "projects" / "nairobi_v2"

INPUT_CSV = (
    PROJECT
    / "results/final_mapping_v2/seed_42/"
      "21a_full_area_mapping_dataset_final_v2/"
      "full_area_mapping_cellwise_final_v2.csv"
)

OUT_DIR = (
    PROJECT
    / "results/final_mapping_v2/seed_42/"
      "21c_publication_spatial_figures_v2"
)

MAIN_PNG = OUT_DIR / "figure_final_mapping_main_v2.png"
MAIN_PDF = OUT_DIR / "figure_final_mapping_main_v2.pdf"
MAIN_TIF = OUT_DIR / "figure_final_mapping_main_v2.tif"

SUPP_S2_PNG = OUT_DIR / "figure_s2_vs_final_supp_v2.png"
SUPP_S2_PDF = OUT_DIR / "figure_s2_vs_final_supp_v2.pdf"

SUPP_CHANGE_PNG = OUT_DIR / "figure_s2_to_final_change_supp_v2.png"
SUPP_CHANGE_PDF = OUT_DIR / "figure_s2_to_final_change_supp_v2.pdf"

AUDIT_JSON = OUT_DIR / "21c_publication_spatial_figures_audit_v2.json"

EXPECTED_N = 112600
EXPECTED_CHANGED = 1557
EXPECTED_CHANGED_0_TO_1 = 593
EXPECTED_CHANGED_1_TO_0 = 964

# CRS is frozen from the source-raster metadata audit.
SOURCE_EPSG = 3857
DISPLAY_EPSG = 4326

# Publication-oriented palette. Kept identical between reference and prediction.
CLASS_COLORS = ["#d9d9d9", "#d95f0e"]  # non-informal, informal
CHANGE_COLORS = ["#eeeeee", "#1f78b4", "#e31a1c"]  # unchanged, 0->1, 1->0

MAIN_DPI = 600
SUPP_DPI = 600


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    required = {
        "global_cell_row",
        "global_cell_col",
        "label",
        "s2_prediction",
        "final_prediction",
        "s2_to_final_change",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if len(df) != EXPECTED_N:
        raise ValueError(f"Rows {len(df)} != expected {EXPECTED_N}")

    if df.duplicated(["global_cell_row", "global_cell_col"]).any():
        raise ValueError("Duplicate global cells")

    changed = df["s2_prediction"].astype(int) != df["final_prediction"].astype(int)
    if int(changed.sum()) != EXPECTED_CHANGED:
        raise ValueError(
            f"S2->Final changed cells {int(changed.sum())} != {EXPECTED_CHANGED}"
        )

    n01 = int((df["s2_to_final_change"] == "changed_0_to_1").sum())
    n10 = int((df["s2_to_final_change"] == "changed_1_to_0").sum())

    if n01 != EXPECTED_CHANGED_0_TO_1:
        raise ValueError(f"0->1 count {n01} != {EXPECTED_CHANGED_0_TO_1}")
    if n10 != EXPECTED_CHANGED_1_TO_0:
        raise ValueError(f"1->0 count {n10} != {EXPECTED_CHANGED_1_TO_0}")

    # Geometry audit from the now-confirmed 80 m EPSG:3857 support.
    widths = df["xmax"].astype(float) - df["xmin"].astype(float)
    heights = df["ymax"].astype(float) - df["ymin"].astype(float)
    if not np.allclose(widths.to_numpy(), 80.0):
        raise ValueError("Not all cells have 80 m width")
    if not np.allclose(heights.to_numpy(), 80.0):
        raise ValueError("Not all cells have 80 m height")

    return df


def build_grid(df: pd.DataFrame, value_col: str):
    rmin = int(df["global_cell_row"].min())
    rmax = int(df["global_cell_row"].max())
    cmin = int(df["global_cell_col"].min())
    cmax = int(df["global_cell_col"].max())

    grid = np.full(
        (rmax - rmin + 1, cmax - cmin + 1),
        np.nan,
        dtype=float,
    )

    rr = df["global_cell_row"].to_numpy(dtype=int) - rmin
    cc = df["global_cell_col"].to_numpy(dtype=int) - cmin
    vv = df[value_col].to_numpy(dtype=float)
    grid[rr, cc] = vv

    extent = [
        float(df["xmin"].min()),
        float(df["xmax"].max()),
        float(df["ymin"].min()),
        float(df["ymax"].max()),
    ]

    return grid, extent


def geographic_formatters():
    transformer = Transformer.from_crs(
        f"EPSG:{SOURCE_EPSG}",
        f"EPSG:{DISPLAY_EPSG}",
        always_xy=True,
    )

    def fmt_lon(x, _pos):
        lon, _ = transformer.transform(x, 0.0)
        return f"{lon:.2f}°E"

    def fmt_lat(y, _pos):
        _, lat = transformer.transform(0.0, y)
        if lat < 0:
            return f"{abs(lat):.2f}°S"
        return f"{lat:.2f}°N"

    return FuncFormatter(fmt_lon), FuncFormatter(fmt_lat)


def style_map_axis(ax, extent, panel_label, title):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    xfmt, yfmt = geographic_formatters()
    ax.xaxis.set_major_formatter(xfmt)
    ax.yaxis.set_major_formatter(yfmt)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, pad=8)

    if panel_label:
        ax.text(
            0.015,
            0.985,
            panel_label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            fontweight="bold",
        )

    # 10 km scale bar. EPSG:3857 uses metre units.
    scale_m = 10000.0
    x0 = extent[0] + 0.06 * (extent[1] - extent[0])
    y0 = extent[2] + 0.055 * (extent[3] - extent[2])
    ax.plot([x0, x0 + scale_m], [y0, y0], lw=2.2, color="black")
    ax.plot([x0, x0], [y0 - 250, y0 + 250], lw=1.4, color="black")
    ax.plot([x0 + scale_m, x0 + scale_m], [y0 - 250, y0 + 250], lw=1.4, color="black")
    ax.text(
        x0 + scale_m / 2,
        y0 + 600,
        "10 km",
        ha="center",
        va="bottom",
        fontsize=9,
    )


def draw_binary_map(ax, grid, extent, panel_label, title):
    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad("white")

    ax.imshow(
        grid,
        extent=extent,
        origin="upper",
        interpolation="none",
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="equal",
    )
    style_map_axis(ax, extent, panel_label, title)


def draw_change_map(ax, grid, extent, panel_label, title):
    cmap = ListedColormap(CHANGE_COLORS)
    cmap.set_bad("white")

    ax.imshow(
        grid,
        extent=extent,
        origin="upper",
        interpolation="none",
        cmap=cmap,
        vmin=0,
        vmax=2,
        aspect="equal",
    )
    style_map_axis(ax, extent, panel_label, title)


def save_main(reference_grid, final_grid, extent):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 5.8),
        constrained_layout=True,
    )

    draw_binary_map(
        axes[0],
        reference_grid,
        extent,
        "(a)",
        "Reference map",
    )
    draw_binary_map(
        axes[1],
        final_grid,
        extent,
        "(b)",
        "Final prediction",
    )

    legend_handles = [
        Patch(facecolor=CLASS_COLORS[0], edgecolor="none", label="Non-informal areas"),
        Patch(facecolor=CLASS_COLORS[1], edgecolor="none", label="Informal settlement"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.savefig(MAIN_PNG, dpi=MAIN_DPI, bbox_inches="tight")
    fig.savefig(MAIN_PDF, bbox_inches="tight")
    fig.savefig(MAIN_TIF, dpi=MAIN_DPI, bbox_inches="tight")
    plt.close(fig)


def save_s2_vs_final(s2_grid, final_grid, extent):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 5.8),
        constrained_layout=True,
    )

    draw_binary_map(
        axes[0],
        s2_grid,
        extent,
        "(a)",
        "S2-only prediction",
    )
    draw_binary_map(
        axes[1],
        final_grid,
        extent,
        "(b)",
        "Final prediction",
    )

    legend_handles = [
        Patch(facecolor=CLASS_COLORS[0], edgecolor="none", label="Non-informal areas"),
        Patch(facecolor=CLASS_COLORS[1], edgecolor="none", label="Informal settlement"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.savefig(SUPP_S2_PNG, dpi=SUPP_DPI, bbox_inches="tight")
    fig.savefig(SUPP_S2_PDF, bbox_inches="tight")
    plt.close(fig)


def save_change(change_grid, extent):
    fig, ax = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)

    draw_change_map(
        ax,
        change_grid,
        extent,
        None,
        "Changes from S2-only to final predictions",
    )

    legend_handles = [
        Patch(facecolor=CHANGE_COLORS[0], edgecolor="none", label="Unchanged"),
        Patch(
            facecolor=CHANGE_COLORS[1],
            edgecolor="none",
            label="Non-informal → Informal",
        ),
        Patch(
            facecolor=CHANGE_COLORS[2],
            edgecolor="none",
            label="Informal → Non-informal",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        fontsize=9,
    )

    fig.savefig(SUPP_CHANGE_PNG, dpi=SUPP_DPI, bbox_inches="tight")
    fig.savefig(SUPP_CHANGE_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    protected = [
        MAIN_PNG,
        MAIN_PDF,
        MAIN_TIF,
        SUPP_S2_PNG,
        SUPP_CHANGE_PNG,
        AUDIT_JSON,
    ]
    existing = [p for p in protected if p.exists()]
    if existing:
        raise FileExistsError(
            "Protected Step21C outputs already exist; refusing overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )

    df = load_data()

    reference_grid, extent = build_grid(df, "label")
    s2_grid, extent_s2 = build_grid(df, "s2_prediction")
    final_grid, extent_final = build_grid(df, "final_prediction")

    if extent_s2 != extent or extent_final != extent:
        raise ValueError("Extent mismatch among publication grids")

    change_code = np.zeros(len(df), dtype=np.int8)
    change_code[df["s2_to_final_change"].eq("changed_0_to_1").to_numpy()] = 1
    change_code[df["s2_to_final_change"].eq("changed_1_to_0").to_numpy()] = 2

    change_df = df.copy()
    change_df["change_code"] = change_code
    change_grid, extent_change = build_grid(change_df, "change_code")
    if extent_change != extent:
        raise ValueError("Change-map extent mismatch")

    save_main(reference_grid, final_grid, extent)
    save_s2_vs_final(s2_grid, final_grid, extent)
    save_change(change_grid, extent)

    audit = {
        "status": "PASSED",
        "step": "21c",
        "purpose": "Publication-quality final spatial figures from frozen Step21A outputs.",
        "new_model_training": False,
        "new_model_inference": False,
        "new_thresholding": False,
        "spatial_smoothing_or_interpolation": False,
        "source_crs_epsg": SOURCE_EPSG,
        "display_coordinates": "WGS84-formatted longitude/latitude tick labels",
        "cell_support": "Original 80 m cells",
        "n_valid_cells": int(len(df)),
        "n_s2_to_final_changed": int(
            (df["s2_prediction"].astype(int) != df["final_prediction"].astype(int)).sum()
        ),
        "main_figure": {
            "panel_a": "Reference labels",
            "panel_b": "Final conditional-routing hard prediction",
            "png": str(MAIN_PNG),
            "pdf": str(MAIN_PDF),
            "tif": str(MAIN_TIF),
        },
        "supplementary_candidates": {
            "s2_vs_final_png": str(SUPP_S2_PNG),
            "change_map_png": str(SUPP_CHANGE_PNG),
        },
        "interpretation_boundary": (
            "The main full-area figure is a descriptive mapping product. "
            "Independent predictive-performance claims remain based on the frozen held-out Test."
        ),
    }

    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("=" * 92)
    print("Nairobi v2 — Step21C Publication Spatial Figures")
    print("=" * 92)
    print("STATUS: PASSED")
    print("CRS: EPSG:3857 confirmed from source-raster metadata")
    print("Display: longitude/latitude tick labels derived from EPSG:3857")
    print("Valid cells:", len(df))
    print("S2 -> Final changed cells:", EXPECTED_CHANGED)
    print()
    print("Main figure:")
    print(" ", MAIN_PNG)
    print(" ", MAIN_PDF)
    print(" ", MAIN_TIF)
    print()
    print("Supplementary candidates:")
    print(" ", SUPP_S2_PNG)
    print(" ", SUPP_CHANGE_PNG)
    print()
    print("Audit:")
    print(" ", AUDIT_JSON)


if __name__ == "__main__":
    main()
