#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

PROJECT = Path.home() / "projects" / "nairobi_v2"
STEP18_MEDIAN = PROJECT / "results/xai_gradcam_v2/seed_42/representative_median_v2/representative_median_metrics_v2.csv"
STEP18_HIGH = PROJECT / "results/xai_gradcam_v2/seed_42/representative_high_impact_v2/representative_high_impact_metrics_v2.csv"
STEP19B = PROJECT / "results/xai_semantic_v2/seed_42/19b_coarse_semantic_grouping"
FRACTIONS_B = STEP19B / "coarse_semantic_fractions_final_v2.csv"
COARSE_OVERLAY_DIR = STEP19B / "coarse_segmentation_overlay"
STEP19C = PROJECT / "results/xai_semantic_v2/seed_42/19c_semantic_gradcam_quantification"
CASEWISE_C = STEP19C / "semantic_gradcam_casewise_final_v2.csv"
OUT_DIR = PROJECT / "results/xai_semantic_v2/seed_42/19d_publication_figures"
MAIN_SEMANTIC = OUT_DIR / "figure_semantic_gradcam_median_main_final"
MAIN_ENRICH = OUT_DIR / "figure_semantic_enrichment_median_main_final"
SUPP_SEMANTIC = OUT_DIR / "figure_semantic_gradcam_high_impact_supp_final"
SUPP_ENRICH = OUT_DIR / "figure_semantic_enrichment_high_impact_supp_final"
AUDIT_JSON = OUT_DIR / "publication_figures_audit_final_v2.json"
NOTES_TXT = OUT_DIR / "publication_figures_notes_final_v2.txt"

SUBTYPE_ALIASES = {
    "fp_to_correct": "fp_to_correct", "s2_fp_to_correct": "fp_to_correct",
    "fn_to_correct": "fn_to_correct", "s2_fn_to_correct": "fn_to_correct",
    "correct_to_fp": "correct_to_fp", "s2_correct_to_fusion_fp": "correct_to_fp",
    "correct_to_fn": "correct_to_fn", "s2_correct_to_fusion_fn": "correct_to_fn",
}
SUBTYPE_ORDER = ["fp_to_correct", "fn_to_correct", "correct_to_fp", "correct_to_fn"]
SUBTYPE_LABEL = {
    "fp_to_correct": "S2 FP →\nFusion correct",
    "fn_to_correct": "S2 FN →\nFusion correct",
    "correct_to_fp": "S2 correct →\nFusion FP",
    "correct_to_fn": "S2 correct →\nFusion FN",
}
SUBTYPE_SHORT = {"fp_to_correct": "FP→C", "fn_to_correct": "FN→C", "correct_to_fp": "C→FP", "correct_to_fn": "C→FN"}
CORE_GROUPS = ["sky", "built_structure", "road_surface", "bare_ground", "vegetation"]
ALL_GROUPS = ["sky", "built_structure", "road_surface", "bare_ground", "vegetation", "dynamic_objects", "other_uncertain"]
GROUP_LABEL = {
    "sky": "Sky", "built_structure": "Built structure", "road_surface": "Road surface",
    "bare_ground": "Bare ground", "vegetation": "Vegetation",
    "dynamic_objects": "Dynamic objects", "other_uncertain": "Other/uncertain",
}
GROUP_SHORT = {
    "sky": "Sky", "built_structure": "Built", "road_surface": "Road", "bare_ground": "Bare",
    "vegetation": "Veg", "dynamic_objects": "Dynamic", "other_uncertain": "Other",
}
COARSE_RGB = {
    "sky": (145, 190, 225), "built_structure": (203, 111, 91), "road_surface": (125, 129, 136),
    "bare_ground": (194, 154, 100), "vegetation": (93, 154, 93),
    "dynamic_objects": (139, 119, 170), "other_uncertain": (190, 130, 170),
}
GRADCAM_CMAP = "viridis"
GRADCAM_ALPHA = 0.45
ENRICH_CMAP = "RdBu_r"
FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
PNG_DPI = 300
TIFF_DPI = 500
FIG_WIDTH_MM = 190.0
SEMANTIC_HEIGHT_MM = 118.0
ENRICH_HEIGHT_MM = 96.0
ROW_LABEL_MM = 22.0
IMAGE_MM = 37.0
WITHIN_PAIR_GAP_MM = 2.5
BETWEEN_K_GAP_MM = 5.0
PANORAMA_HEIGHT_MM = IMAGE_MM / (500.0 / 128.0)
BASE_PT = 7.2
GROUP_PT = 9.0
HEADER_PT = 7.8
ROW_PT = 7.0
ANNOTATION_PT = 6.15
LEGEND_PT = 6.15
QUANT_PT = 7.0
PANEL_PT = 8.2


def canonical_subtype(x):
    s = str(x).strip().lower()
    if s not in SUBTYPE_ALIASES:
        raise ValueError(f"Unknown subtype: {x}")
    return SUBTYPE_ALIASES[s]


def canonical_k(x):
    s = str(x).replace(" ", "")
    if s in {"K1", "K=1", "1"}:
        return "K1"
    if s in {"K>=2", "K≥2", ">=2", "≥2"}:
        return "K>=2"
    raise ValueError(f"Unknown K stratum: {x}")


def resolve_existing_path(value):
    p = Path(str(value))
    if p.is_file():
        return p
    marker = "projects/nairobi_v2/"
    s = str(p)
    if marker in s:
        q = PROJECT / s.split(marker, 1)[1]
        if q.is_file():
            return q
    raise FileNotFoundError(value)


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT_FAMILY, "font.size": BASE_PT,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.linewidth": 0.65, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def save_all(fig, stem):
    pdf, png, tif = stem.with_suffix(".pdf"), stem.with_suffix(".png"), stem.with_suffix(".tif")
    fig.savefig(pdf, bbox_inches=None, facecolor="white")
    fig.savefig(png, dpi=PNG_DPI, bbox_inches=None, facecolor="white")
    fig.savefig(tif, dpi=TIFF_DPI, bbox_inches=None, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    return {"pdf": str(pdf), "png": str(png), "tif": str(tif)}


def coarse_color(group):
    return tuple(v / 255.0 for v in COARSE_RGB[group])


def load_rgb(path):
    arr = np.asarray(Image.open(path).convert("RGB"))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected RGB shape {arr.shape}: {path}")
    return arr


def load_heatmap(path):
    arr = np.squeeze(np.asarray(np.load(path, allow_pickle=False), dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"Grad-CAM must be 2-D: {path}, shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite Grad-CAM values: {path}")
    if float(arr.min()) < -1e-5:
        raise ValueError(f"Negative Grad-CAM values: {path}, min={float(arr.min())}")
    return np.clip(arr, 0.0, 1.0)


def clean_image_axis(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def canonicalize_frame(df, selection=None):
    out = df.copy()
    if selection is not None:
        out["selection"] = selection
    if "selection" not in out.columns:
        raise ValueError("Missing selection column")
    out["selection"] = out["selection"].astype(str).str.strip()
    out["subtype"] = out["subtype"].map(canonical_subtype)
    out["k_stratum"] = out["k_stratum"].map(canonical_k)
    out["covered_order"] = pd.to_numeric(out["covered_order"]).astype(int)
    out["target_embedding_row"] = pd.to_numeric(out["target_embedding_row"]).astype(int)
    return out


def dataframe_case_key(df):
    return (df["selection"].astype(str) + "|" + df["subtype"].astype(str) + "|" + df["k_stratum"].astype(str)
            + "|" + df["covered_order"].astype(str) + "|" + df["target_embedding_row"].astype(str))


def coarse_overlay_path(row):
    kcode = "k1" if row["k_stratum"] == "K1" else "kge2"
    stem = (f'{row["selection"]}__{row["subtype"]}__{kcode}__order{int(row["covered_order"])}'
            f'__emb{int(row["target_embedding_row"])}__coarse_overlay.png')
    p = COARSE_OVERLAY_DIR / stem
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_inputs():
    med18 = canonicalize_frame(pd.read_csv(STEP18_MEDIAN), "median")
    high18 = canonicalize_frame(pd.read_csv(STEP18_HIGH), "high_impact")
    s18 = pd.concat([med18, high18], ignore_index=True)
    fractions = canonicalize_frame(pd.read_csv(FRACTIONS_B))
    fractions["area_fraction"] = pd.to_numeric(fractions["area_fraction"], errors="raise")
    casewise = canonicalize_frame(pd.read_csv(CASEWISE_C))
    for col in ["area_fraction", "gradcam_response_fraction", "enrichment", "log2_enrichment"]:
        casewise[col] = pd.to_numeric(casewise[col], errors="coerce")
    validate_inputs(s18, fractions, casewise)
    return s18, fractions, casewise


def validate_inputs(s18, fractions, casewise):
    if len(s18) != 16:
        raise ValueError(f"Step18 expected 16 rows, found {len(s18)}")
    expected = {(s, k) for s in SUBTYPE_ORDER for k in ["K1", "K>=2"]}
    for selection in ["median", "high_impact"]:
        d = s18[s18["selection"] == selection]
        if len(d) != 8:
            raise ValueError(f"{selection}: expected 8 Step18 cases, found {len(d)}")
        observed = set(zip(d["subtype"], d["k_stratum"]))
        if observed != expected:
            raise ValueError(f"{selection}: subtype/K structure mismatch")

    if len(fractions) != 112:
        raise ValueError(f"19b expected 112 rows, found {len(fractions)}")
    gcols = ["selection", "subtype", "k_stratum", "covered_order", "target_embedding_row"]
    frac_counts = fractions.groupby(gcols)["coarse_group"].nunique()
    if not (frac_counts == 7).all():
        raise ValueError("19b must contain all seven coarse groups for every case")
    frac_sums = fractions.groupby(gcols)["area_fraction"].sum()
    if not np.allclose(frac_sums.values, 1.0, atol=1e-8):
        raise ValueError(f"19b area fractions do not conserve to 1: {frac_sums.min()}..{frac_sums.max()}")

    if len(casewise) != 112:
        raise ValueError(f"19c expected 112 rows, found {len(casewise)}")
    c_counts = casewise.groupby(gcols)["coarse_group"].nunique()
    if not (c_counts == 7).all():
        raise ValueError("19c must contain all seven coarse groups for every case")
    response_sums = casewise.groupby(gcols)["gradcam_response_fraction"].sum()
    if not np.allclose(response_sums.values, 1.0, atol=1e-6):
        raise ValueError(f"19c response fractions do not conserve to 1: {response_sums.min()}..{response_sums.max()}")

    s18_keys = set(dataframe_case_key(s18))
    bkeys = fractions[gcols].drop_duplicates(); ckeys = casewise[gcols].drop_duplicates()
    if s18_keys != set(dataframe_case_key(bkeys)) or s18_keys != set(dataframe_case_key(ckeys)):
        raise ValueError("Case-key mismatch across Step18 / Step19b / Step19c")

    key = gcols + ["coarse_group"]
    joined = fractions[key + ["area_fraction"]].merge(
        casewise[key + ["area_fraction"]], on=key, suffixes=("_19b", "_19c"), validate="one_to_one")
    if not np.allclose(joined["area_fraction_19b"], joined["area_fraction_19c"], atol=1e-10):
        raise ValueError("19b and 19c area fractions are not identical")

    for _, r in s18.iterrows():
        rgb_path = resolve_existing_path(r["original_png"]); heat_path = resolve_existing_path(r["heatmap_npy"])
        rgb = load_rgb(rgb_path); heat = load_heatmap(heat_path); sem = load_rgb(coarse_overlay_path(r))
        if rgb.shape[:2] != (128, 500):
            raise ValueError(f"Step18 RGB must be 128×500, got {rgb.shape[:2]}: {rgb_path}")
        if heat.shape != (128, 500):
            raise ValueError(f"Step18 Grad-CAM must be 128×500, got {heat.shape}: {heat_path}")
        if sem.shape[:2] != (128, 500):
            raise ValueError(f"19b coarse overlay must be 128×500, got {sem.shape[:2]}: {coarse_overlay_path(r)}")


def top3_text(fractions, row):
    mask = ((fractions["selection"] == row["selection"]) & (fractions["subtype"] == row["subtype"])
            & (fractions["k_stratum"] == row["k_stratum"]) & (fractions["covered_order"] == int(row["covered_order"]))
            & (fractions["target_embedding_row"] == int(row["target_embedding_row"])))
    g = fractions.loc[mask, ["coarse_group", "area_fraction"]].copy()
    if len(g) != 7:
        raise ValueError(f"Expected 7 coarse groups for top-3 annotation, found {len(g)}")
    g = g.sort_values(["area_fraction", "coarse_group"], ascending=[False, True]).head(3)
    parts = [f'{GROUP_SHORT[r.coarse_group]} {100.0*r.area_fraction:.0f}%' for r in g.itertuples(index=False)]
    return "  |  ".join(parts)


def metric_annotation(row):
    target = str(row["target_direction"]).replace("_", "-").strip()
    delta = float(row["delta_gsv_logit"])
    if row["k_stratum"] == "K1":
        return f"Target: {target}  |  ΔGSV = {delta:+.2f}"
    loo = float(row["signed_effect"])
    return f"Target: {target}  |  ΔGSV = {delta:+.2f}  |  LOO = {loo:+.2f}"


def get_step18_case(s18, selection, subtype, k):
    d = s18[(s18["selection"] == selection) & (s18["subtype"] == subtype) & (s18["k_stratum"] == k)]
    if len(d) != 1:
        raise ValueError(f"Expected exactly one Step18 case for {selection}/{subtype}/{k}, found {len(d)}")
    return d.iloc[0]


def validate_text_inside_figure(fig, tolerance_px=3.0):
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); fb = fig.bbox; bad = []
    for obj in fig.findobj(match=matplotlib.text.Text):
        if not obj.get_visible() or not str(obj.get_text()).strip():
            continue
        bb = obj.get_window_extent(renderer=renderer)
        if bb.x0 < fb.x0-tolerance_px or bb.y0 < fb.y0-tolerance_px or bb.x1 > fb.x1+tolerance_px or bb.y1 > fb.y1+tolerance_px:
            bad.append(str(obj.get_text()))
    if bad:
        raise RuntimeError("Publication text clipping detected:\n" + "\n".join(bad))


def build_semantic_gradcam_panel(selection, s18, fractions, stem):
    setup_matplotlib()

    fig = plt.figure(
        figsize=(FIG_WIDTH_MM / 25.4, SEMANTIC_HEIGHT_MM / 25.4),
        constrained_layout=False,
    )

    widths = [
        ROW_LABEL_MM,
        3.0,
        IMAGE_MM,
        WITHIN_PAIR_GAP_MM,
        IMAGE_MM,
        BETWEEN_K_GAP_MM,
        IMAGE_MM,
        WITHIN_PAIR_GAP_MM,
        IMAGE_MM,
    ]

    outer = fig.add_gridspec(
        nrows=11,
        ncols=1,
        height_ratios=[
            4.8,    # K headers
            4.2,    # column headers
            14.8,   # case row 1
            2.6,    # gap
            14.8,   # case row 2
            2.6,    # gap
            14.8,   # case row 3
            2.6,    # gap
            14.8,   # case row 4
            5.6,    # semantic legend
            8.4,    # Grad-CAM title + colorbar
        ],
        left=3.0 / FIG_WIDTH_MM,
        right=1.0 - 3.0 / FIG_WIDTH_MM,
        bottom=0.040,
        top=0.98,
        hspace=0.0,
    )

    # K headers
    hg = outer[0].subgridspec(
        1, 9, width_ratios=widths, wspace=0.0
    )

    ax = fig.add_subplot(hg[0, 2:5])
    ax.axis("off")
    ax.text(
        0.5, 0.48, "K = 1",
        ha="center", va="center",
        fontsize=GROUP_PT, fontweight="bold",
    )

    ax = fig.add_subplot(hg[0, 6:9])
    ax.axis("off")
    ax.text(
        0.5, 0.48, "K ≥ 2",
        ha="center", va="center",
        fontsize=GROUP_PT, fontweight="bold",
    )

    # Column headers
    ch = outer[1].subgridspec(
        1, 9, width_ratios=widths, wspace=0.0
    )

    for col, label in [
        (2, "Coarse semantic overlay"),
        (4, "Directional Grad-CAM"),
        (6, "Coarse semantic overlay"),
        (8, "Directional Grad-CAM"),
    ]:
        ax = fig.add_subplot(ch[0, col])
        ax.axis("off")
        ax.text(
            0.5, 0.5, label,
            ha="center", va="center",
            fontsize=HEADER_PT,
        )

    # Frozen representative cases
    for i, (subtype, outer_row) in enumerate(
        zip(SUBTYPE_ORDER, [2, 4, 6, 8])
    ):
        rg = outer[outer_row].subgridspec(
            2,
            9,
            height_ratios=[PANORAMA_HEIGHT_MM, 3.7],
            width_ratios=widths,
            hspace=0.0,
            wspace=0.0,
        )

        ax_label = fig.add_subplot(rg[:, 0])
        ax_label.axis("off")
        ax_label.text(
            0.03,
            0.56,
            f"({chr(97+i)}) {SUBTYPE_LABEL[subtype]}",
            ha="left",
            va="center",
            fontsize=ROW_PT,
            fontweight="bold",
            linespacing=1.05,
        )

        for k, sem_col, cam_col, ann_slice in [
            ("K1", 2, 4, slice(2, 5)),
            ("K>=2", 6, 8, slice(6, 9)),
        ]:
            row = get_step18_case(
                s18, selection, subtype, k
            )

            rgb = load_rgb(
                resolve_existing_path(row["original_png"])
            )
            sem = load_rgb(
                coarse_overlay_path(row)
            )
            heat = load_heatmap(
                resolve_existing_path(row["heatmap_npy"])
            )

            if (
                rgb.shape[:2] != heat.shape
                or sem.shape[:2] != rgb.shape[:2]
            ):
                raise ValueError(
                    "Semantic/RGB/Grad-CAM spatial mismatch"
                )

            ax_sem = fig.add_subplot(rg[0, sem_col])
            ax_sem.imshow(sem)
            clean_image_axis(ax_sem)

            ax_cam = fig.add_subplot(rg[0, cam_col])
            ax_cam.imshow(rgb)
            ax_cam.imshow(
                heat,
                cmap=GRADCAM_CMAP,
                alpha=GRADCAM_ALPHA,
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            clean_image_axis(ax_cam)

            # Step 19 contributes the semantic-composition annotation only.
            # Target / DeltaGSV / LOO remain in the Step-18 explanatory panel
            # and are deliberately not repeated here.
            ax_ann = fig.add_subplot(rg[1, ann_slice])
            ax_ann.axis("off")
            ax_ann.text(
                0.5,
                0.50,
                top3_text(fractions, row),
                ha="center",
                va="center",
                fontsize=ANNOTATION_PT,
            )

    # Fixed Step-19b semantic palette legend
    lg = outer[9].subgridspec(
        1, 9, width_ratios=widths, wspace=0.0
    )
    ax_leg = fig.add_subplot(lg[0, 2:9])
    ax_leg.axis("off")

    handles = [
        Patch(
            facecolor=coarse_color(g),
            edgecolor="none",
            label=GROUP_LABEL[g],
        )
        for g in ALL_GROUPS
    ]

    ax_leg.legend(
        handles=handles,
        loc="center",
        ncol=7,
        frameon=False,
        fontsize=LEGEND_PT,
        handlelength=1.15,
        handleheight=0.8,
        columnspacing=1.05,
        borderaxespad=0.0,
        labelspacing=0.3,
    )

    # Grad-CAM legend: exact naming retained from finalized Step 18.
    # Title and Low/High ticks have dedicated, non-overlapping bands.
    cb_outer = outer[10].subgridspec(
        2,
        9,
        height_ratios=[2.6, 4.2],
        width_ratios=widths,
        hspace=0.18,
        wspace=0.0,
    )

    ax_cb_title = fig.add_subplot(cb_outer[0, 2:9])
    ax_cb_title.axis("off")
    ax_cb_title.text(
        0.5,
        0.46,
        "Relative Grad-CAM response",
        ha="center",
        va="center",
        fontsize=LEGEND_PT,
    )

    cb_sub = cb_outer[1, 2:9].subgridspec(
        1,
        3,
        width_ratios=[48.5, 62.0, 48.5],
        wspace=0.0,
    )

    cax = fig.add_subplot(cb_sub[0, 1])
    sm = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap=GRADCAM_CMAP,
    )
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
    )
    cb.set_ticks([0.0, 1.0])
    cb.set_ticklabels(["Low", "High"])
    cb.ax.tick_params(
        labelsize=LEGEND_PT,
        length=2.2,
        width=0.6,
        pad=1.2,
    )
    cb.outline.set_linewidth(0.6)

    validate_text_inside_figure(fig)
    outputs = save_all(fig, stem)
    plt.close(fig)
    return outputs


def shared_enrichment_limit(casewise):
    vals = casewise.loc[casewise["coarse_group"].isin(CORE_GROUPS),"log2_enrichment"].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return max(1.0, np.ceil(float(np.max(np.abs(vals)))*2.0)/2.0)


def selection_enrichment_matrix(casewise, selection):
    rows = casewise[(casewise["selection"] == selection) & (casewise["coarse_group"].isin(CORE_GROUPS))].copy()
    keys = []
    for subtype in SUBTYPE_ORDER:
        for k in ["K1","K>=2"]:
            d = rows[(rows["subtype"] == subtype) & (rows["k_stratum"] == k)][["covered_order","target_embedding_row"]].drop_duplicates()
            if len(d) != 1:
                raise ValueError(f"{selection}/{subtype}/{k}: expected one case, found {len(d)}")
            rr = d.iloc[0]; keys.append((subtype,k,int(rr["covered_order"]),int(rr["target_embedding_row"])))
    mat = np.full((8,len(CORE_GROUPS)),np.nan,float)
    for i,(subtype,k,order,emb) in enumerate(keys):
        d = rows[(rows["subtype"]==subtype)&(rows["k_stratum"]==k)&(rows["covered_order"]==order)&(rows["target_embedding_row"]==emb)]
        for j,group in enumerate(CORE_GROUPS):
            g = d[d["coarse_group"]==group]
            if len(g) != 1:
                raise ValueError(f"{selection}/{subtype}/{k}/{group}: expected 1 row, found {len(g)}")
            mat[i,j] = float(g.iloc[0]["log2_enrichment"])
    return keys, mat, rows


def build_enrichment_figure(
    selection,
    casewise,
    shared_vlim,
    stem,
):
    setup_matplotlib()
    keys, mat, rows = selection_enrichment_matrix(
        casewise, selection
    )

    cmap = plt.get_cmap(ENRICH_CMAP).copy()
    cmap.set_bad("#f2f2f2")
    norm = TwoSlopeNorm(
        vmin=-shared_vlim,
        vcenter=0.0,
        vmax=shared_vlim,
    )

    fig = plt.figure(
        figsize=(FIG_WIDTH_MM / 25.4, ENRICH_HEIGHT_MM / 25.4),
        constrained_layout=False,
    )

    # Explicit spacer column prevents Panel-B y labels from intruding
    # into Panel A. A dedicated bottom band under Panel B contains the
    # legend and the zero-reference explanation, so neither overlaps data.
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.28, 0.46, 1.0],
        height_ratios=[1.0, 0.18],
        left=0.105,
        right=0.975,
        top=0.92,
        bottom=0.13,
        wspace=0.0,
        hspace=0.22,
    )

    # -----------------------------------------------------------------
    # Panel (a): case-level heatmap
    # -----------------------------------------------------------------
    ax_h = fig.add_subplot(gs[0, 0])
    ax_h.imshow(
        np.ma.masked_invalid(mat),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax_h.set_xticks(np.arange(5))
    ax_h.set_xticklabels(
        [
            "Sky",
            "Built\nstructure",
            "Road\nsurface",
            "Bare\nground",
            "Vegetation",
        ],
        fontsize=QUANT_PT,
    )

    ax_h.set_yticks(np.arange(8))
    ax_h.set_yticklabels(
        [
            f"{SUBTYPE_SHORT[s]} | "
            f"{'K=1' if k == 'K1' else 'K≥2'}"
            for s, k, _, _ in keys
        ],
        fontsize=QUANT_PT,
    )
    ax_h.tick_params(length=0)

    ax_h.set_xticks(
        np.arange(-0.5, 5, 1),
        minor=True,
    )
    ax_h.set_yticks(
        np.arange(-0.5, 8, 1),
        minor=True,
    )
    ax_h.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.55,
    )
    ax_h.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i, j]):
                ax_h.text(
                    j,
                    i,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="#444444",
                )

    ax_h.text(
        -0.12,
        1.035,
        "(a)",
        transform=ax_h.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_PT,
        fontweight="bold",
    )

    cax = fig.add_subplot(gs[1, 0])
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )

    display_ticks = np.linspace(
        -shared_vlim,
        shared_vlim,
        5,
    )
    cb.set_ticks(display_ticks)
    cb.set_label(
        "log₂ enrichment",
        fontsize=QUANT_PT,
        labelpad=2.0,
    )
    cb.ax.tick_params(
        labelsize=6.5,
        length=2.2,
        width=0.6,
        pad=1.0,
    )
    cb.outline.set_linewidth(0.6)

    # -----------------------------------------------------------------
    # Panel (b): individual cases + median + IQR
    # -----------------------------------------------------------------
    ax_s = fig.add_subplot(gs[0, 2])

    ax_s.axvline(
        0.0,
        color="#555555",
        linewidth=0.75,
        linestyle="--",
        zorder=0,
    )

    # Shape, rather than size, distinguishes individual observations
    # from the median.
    INDIVIDUAL_SIZE = 20.0
    MEDIAN_SIZE = 20.0

    # IQR remains deliberately light and gains short Q1/Q3 end caps so
    # short IQRs remain visible even when the median marker lies nearby.
    IQR_LINEWIDTH = 0.80
    IQR_CAP_HALF_HEIGHT = 0.045

    def collision_aware_offsets(values):
        """
        Deterministic beeswarm-lite offsets.

        Only x-nearby observations are moved vertically. The vertical
        displacement is purely anti-overlap and has no scientific meaning.
        """
        values = np.asarray(values, dtype=float)
        n = len(values)
        offsets = np.zeros(n, dtype=float)

        if n <= 1:
            return offsets

        order = np.argsort(values, kind="mergesort")

        # About 4% of the shared +/- display range. This is wide enough
        # to identify markers that would visually collide at publication size.
        x_threshold = max(0.12, 0.04 * shared_vlim)

        clusters = []
        current = [int(order[0])]

        for prev_idx, idx in zip(order[:-1], order[1:]):
            if abs(values[int(idx)] - values[int(prev_idx)]) <= x_threshold:
                current.append(int(idx))
            else:
                clusters.append(current)
                current = [int(idx)]
        clusters.append(current)

        for cluster in clusters:
            m = len(cluster)
            if m == 1:
                offsets[cluster[0]] = 0.0
                continue

            # Symmetric compact offsets, capped so points remain visibly
            # part of the same semantic-group row.
            half = min(0.12, 0.040 * (m - 1))
            levels = np.linspace(-half, half, m)

            # Assign offsets in stable x order for full reproducibility.
            for idx, off in zip(cluster, levels):
                offsets[idx] = off

        return offsets

    for yi, group in enumerate(CORE_GROUPS):
        vals = rows.loc[
            rows["coarse_group"] == group,
            "log2_enrichment",
        ].to_numpy(float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            continue

        q1, med, q3 = np.quantile(
            vals,
            [0.25, 0.50, 0.75],
        )

        offsets = collision_aware_offsets(vals)
        color = coarse_color(group)

        # Individual representative cases.
        ax_s.scatter(
            vals,
            yi + offsets,
            s=INDIVIDUAL_SIZE,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.40,
            alpha=0.76,
            zorder=2,
        )

        # IQR centre line: exact same y-centre as the median marker.
        ax_s.plot(
            [q1, q3],
            [yi, yi],
            color=color,
            linewidth=IQR_LINEWIDTH,
            solid_capstyle="butt",
            zorder=3,
        )

        # Q1/Q3 end caps make even short IQRs readable.
        ax_s.vlines(
            [q1, q3],
            yi - IQR_CAP_HALF_HEIGHT,
            yi + IQR_CAP_HALF_HEIGHT,
            color=color,
            linewidth=IQR_LINEWIDTH,
            zorder=3,
        )

        # Median marker: same visual area as individual circles; the
        # diamond shape alone identifies the median.
        ax_s.scatter(
            [med],
            [yi],
            s=MEDIAN_SIZE,
            marker="D",
            facecolor=color,
            edgecolor="black",
            linewidth=0.55,
            zorder=4,
        )

    n_labels = []
    for group in CORE_GROUPS:
        n = int(
            np.isfinite(
                rows.loc[
                    rows["coarse_group"] == group,
                    "log2_enrichment",
                ].to_numpy(float)
            ).sum()
        )
        n_labels.append(
            f"{GROUP_LABEL[group]} (n={n})"
        )

    ax_s.set_yticks(np.arange(5))
    ax_s.set_yticklabels(
        n_labels,
        fontsize=QUANT_PT,
    )
    ax_s.invert_yaxis()

    summary_limit = shared_vlim * 1.06
    ax_s.set_xlim(
        -summary_limit,
        summary_limit,
    )
    ax_s.set_xticks(display_ticks)
    ax_s.set_xlabel(
        "log₂ enrichment",
        fontsize=QUANT_PT,
    )

    ax_s.tick_params(
        axis="x",
        labelsize=6.5,
    )
    ax_s.tick_params(
        axis="y",
        length=0,
        pad=4.0,
    )

    ax_s.grid(
        axis="x",
        color="#dddddd",
        linewidth=0.55,
        zorder=0,
    )

    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.spines["left"].set_visible(False)

    ax_s.text(
        -0.15,
        1.035,
        "(b)",
        transform=ax_s.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_PT,
        fontweight="bold",
    )

    # -----------------------------------------------------------------
    # Dedicated bottom band for Panel (b):
    # legend above, zero-reference explanation below.
    # This band is outside the plotting area and therefore cannot cover data.
    # -----------------------------------------------------------------
    bottom_b = gs[1, 2].subgridspec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=0.05,
    )

    ax_leg = fig.add_subplot(bottom_b[0, 0])
    ax_leg.axis("off")

    marker_size_pt = INDIVIDUAL_SIZE ** 0.5
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=marker_size_pt,
            markerfacecolor="#888888",
            markeredgecolor="white",
            markeredgewidth=0.40,
            label="Individual case",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="-",
            linewidth=IQR_LINEWIDTH,
            markersize=marker_size_pt,
            color="#777777",
            markerfacecolor="#999999",
            markeredgecolor="black",
            markeredgewidth=0.55,
            label="Median + IQR",
        ),
    ]

    ax_leg.legend(
        handles=legend_handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=6.2,
        handlelength=2.0,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    ax_note = fig.add_subplot(bottom_b[1, 0])
    ax_note.axis("off")
    ax_note.text(
        0.5,
        0.55,
        "0 = Grad-CAM response proportional to semantic image area",
        ha="center",
        va="center",
        fontsize=6.3,
    )

    validate_text_inside_figure(fig)
    outputs = save_all(fig, stem)
    plt.close(fig)
    return outputs


def main():
    print("="*100); print("Nairobi v2 — Step 19d Final Publication Figure Assembly"); print("="*100)
    print("New segmentation inference: NO\nNew Grad-CAM computation: NO\nNew semantic quantification: NO\nCase reselection: NO\nPurpose: publication-quality figure assembly only\n")
    s18,fractions,casewise = load_inputs()
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    shared_vlim = shared_enrichment_limit(casewise); outputs={}
    print("[1/4] Median semantic + Grad-CAM main figure"); outputs["median_semantic_gradcam"] = build_semantic_gradcam_panel("median",s18,fractions,MAIN_SEMANTIC)
    print("[2/4] Median semantic enrichment main figure"); outputs["median_enrichment"] = build_enrichment_figure("median",casewise,shared_vlim,MAIN_ENRICH)
    print("[3/4] High-impact semantic + Grad-CAM supplementary figure"); outputs["high_impact_semantic_gradcam"] = build_semantic_gradcam_panel("high_impact",s18,fractions,SUPP_SEMANTIC)
    print("[4/4] High-impact semantic enrichment supplementary figure"); outputs["high_impact_enrichment"] = build_enrichment_figure("high_impact",casewise,shared_vlim,SUPP_ENRICH)
    audit = {
        "status":"PASSED","step":"19d","purpose":"publication-quality figure assembly only","scientific_recomputation":False,
        "new_segmentation_inference":False,"gradcam_recomputed":False,"new_semantic_quantification":False,"case_reselection":False,
        "n_cases_total":16,"n_cases_median":8,"n_cases_high_impact":8,"case_key_alignment_step18_19b_19c":True,
        "model_input_grid_px":{"width":500,"height":128},
        "style":{"figure_width_mm":FIG_WIDTH_MM,"semantic_panel_height_mm":SEMANTIC_HEIGHT_MM,"enrichment_panel_height_mm":ENRICH_HEIGHT_MM,
                 "font_family":FONT_FAMILY,"png_dpi":PNG_DPI,"tiff_dpi":TIFF_DPI,"pdf_fonttype":42,"bbox_tight_crop":False,
                 "gradcam_colormap":GRADCAM_CMAP,"gradcam_alpha":GRADCAM_ALPHA,"gradcam_colorbar_label":"Relative Grad-CAM response","gradcam_normalization":"within-image; inherited frozen Step18 heatmaps",
                 "semantic_palette":COARSE_RGB,"semantic_overlay_source":"frozen Step19b coarse_segmentation_overlay files",
                 "enrichment_colormap":ENRICH_CMAP,"enrichment_center":0.0,"enrichment_shared_abs_limit_main_and_supp":shared_vlim,
                 "enrichment_cross_selection_color_scale_comparable":True},
        "main_text":{"semantic_gradcam":str(MAIN_SEMANTIC),"semantic_enrichment":str(MAIN_ENRICH),"selection":"median","core_groups_quantitative":CORE_GROUPS,"top3_area_annotations_use_all_7_groups":True,"top3_area_annotations_renormalized":False,"semantic_case_annotation":"top-3 coarse area fractions only","step18_target_delta_loo_repeated_in_step19":False},
        "supplementary":{"semantic_gradcam":str(SUPP_SEMANTIC),"semantic_enrichment":str(SUPP_ENRICH),"selection":"high_impact"},
        "outputs":outputs,
    }
    AUDIT_JSON.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    NOTES_TXT.write_text(
        "Nairobi v2 — Step 19d final publication figures\n\n"
        "SCIENTIFIC STATUS\nNo new segmentation inference.\nNo Grad-CAM recomputation.\nNo semantic quantification.\nNo case reselection.\n\n"
        "MAIN TEXT\n1) figure_semantic_gradcam_median_main_final\n2) figure_semantic_enrichment_median_main_final\n\n"
        "SUPPLEMENTARY\n3) figure_semantic_gradcam_high_impact_supp_final\n4) figure_semantic_enrichment_high_impact_supp_final\n\n"
        "STYLE CONSISTENCY\nGrad-CAM uses viridis + alpha 0.45, matching finalized Step 18.\n"
        "Semantic overlays are frozen Step 19b outputs and retain the fixed Step 19b palette.\n"
        "Enrichment uses one shared zero-centred RdBu_r scale for median and high-impact figures.\n""Step-19 Grad-CAM colorbar naming is identical to Step 18: Relative Grad-CAM response.\n""Step-19 case annotations contain only the three largest coarse area fractions; Target / DeltaGSV / LOO are not repeated.\n""Enrichment summary uses deterministic collision-aware vertical jitter only where needed to prevent marker overlap.\n""Individual circles and median diamonds use equal marker area; IQR is shown with a thin Q1-Q3 line and short end caps.\n""The enrichment legend is placed in a dedicated band below Panel (b), outside the data region.\n"
        f"PNG={PNG_DPI} dpi; TIFF={TIFF_DPI} dpi (LZW); PDF retains vector text.\n"
        "Exact physical canvas is preserved (no tight bounding-box crop).\n\n"
        "INTERPRETATION\nGrad-CAM is normalized within image; color intensity is not an absolute cross-image magnitude.\n"
        "Top-3 area fractions are selected from all seven coarse groups and are not renormalized.\n"
        "Quantitative main-text plots show only the five core interpretable groups.\n"
        "Undefined enrichment values remain N/A.\n"
        "Panel-b summaries are descriptive for selected representative cases, not population inference.\n",
        encoding="utf-8")
    print("\nSTATUS: PASSED"); print(f"Shared enrichment |log2| display limit: {shared_vlim:.2f}"); print(f"Output: {OUT_DIR}"); print("Audit:",AUDIT_JSON); print("Notes:",NOTES_TXT)


if __name__ == "__main__":
    main()
