#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT = Path.home() / "projects" / "nairobi_v2"
SEED = 42

S2_MODEL = PROJECT / "models/s2_only_v2/seed_42/s2_only_model_best_v2.keras"
X_TRAIN = PROJECT / "scratch/features/s2_inputs/X_train_s2_v2.npy"
S2_VAL_PROBS_FROZEN = PROJECT / "results/s2_only_v2/seed_42/val_probabilities_v2.npy"

LABEL_PATCHES = {
    "train": PROJECT / "metadata/latest_aligned_v2/train_label_patches_v2.npy",
    "val": PROJECT / "metadata/latest_aligned_v2/val_label_patches_v2.npy",
}
VALID_MASKS = {
    "train": PROJECT / "metadata/latest_aligned_v2/train_valid_label_mask_v2.npy",
    "val": PROJECT / "metadata/latest_aligned_v2/val_valid_label_mask_v2.npy",
}

CELL_MAPPING = PROJECT / "metadata/latest_aligned_v2/cell_index_mapping_v2.csv"

STEP20A = (
    PROJECT
    / "results/xai_spatial_v2/seed_42/20a_full_area_delta_gsv_final_v2/"
      "full_area_delta_gsv_cellwise_final_v2.csv"
)

TEST_ALL_VALID = (
    PROJECT / "results/final_test_v2/seed_42/all_valid_test_predictions_v2.csv"
)
TEST_COVERED = (
    PROJECT / "results/final_test_v2/seed_42/covered_test_predictions_v2.csv"
)

S2_THRESHOLD = 0.78
FUSION_THRESHOLD = 0.59

EXPECTED_PATCHES = {"train": 306, "val": 102}
EXPECTED_VALID = {"train": 67220, "val": 23024, "test": 22356, "all": 112600}
EXPECTED_COVERED = {"train": 15481, "val": 4902, "test": 5744, "all": 26127}
EXPECTED_NO_GSV = {
    "train": EXPECTED_VALID["train"] - EXPECTED_COVERED["train"],
    "val": EXPECTED_VALID["val"] - EXPECTED_COVERED["val"],
    "test": EXPECTED_VALID["test"] - EXPECTED_COVERED["test"],
    "all": EXPECTED_VALID["all"] - EXPECTED_COVERED["all"],
}

OUT_DIR = (
    PROJECT
    / "results/final_mapping_v2/seed_42/"
      "21a_full_area_mapping_dataset_final_v2"
)
OUT_CELLWISE = OUT_DIR / "full_area_mapping_cellwise_final_v2.csv"
OUT_SUMMARY = OUT_DIR / "full_area_mapping_summary_final_v2.csv"
OUT_TEST_TRANSITIONS = OUT_DIR / "heldout_test_transition_summary_final_v2.csv"
OUT_AUDIT_JSON = OUT_DIR / "full_area_mapping_audit_final_v2.json"
OUT_AUDIT_TXT = OUT_DIR / "full_area_mapping_audit_final_v2.txt"

os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def parse_bool(series: pd.Series, name: str) -> pd.Series:
    out = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    if out.isna().any():
        raise ValueError(f"Could not parse {name} as boolean")
    return out.astype(bool)


def load_mapping() -> pd.DataFrame:
    df = pd.read_csv(CELL_MAPPING)
    required = {
        "split", "patch_order", "original_patch_index", "row_index_in_split",
        "cell_row", "cell_col", "global_cell_row", "global_cell_col",
        "reference_label", "label", "valid_cell", "informal_fraction",
        "valid_10m_pixel_count", "valid_coverage_fraction",
        "is_padding_or_nodata", "xmin", "ymin", "xmax", "ymax",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cell mapping missing columns: {sorted(missing)}")

    valid = parse_bool(df["valid_cell"], "valid_cell")
    padding = parse_bool(df["is_padding_or_nodata"], "is_padding_or_nodata")
    df = df.loc[valid & (~padding)].copy()
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    if len(df) != EXPECTED_VALID["all"]:
        raise ValueError(f"Valid mapping rows {len(df)} != {EXPECTED_VALID['all']}")

    counts = df["split"].value_counts().to_dict()
    for split in ["train", "val", "test"]:
        if int(counts.get(split, 0)) != EXPECTED_VALID[split]:
            raise ValueError(f"{split} valid mapping count mismatch")

    if df.duplicated(["global_cell_row", "global_cell_col"]).any():
        raise ValueError("Duplicate global cells in mapping")
    if df.duplicated(["split", "patch_order", "cell_row", "cell_col"]).any():
        raise ValueError("Duplicate split/patch/cell addresses in mapping")

    widths = df["xmax"].astype(float) - df["xmin"].astype(float)
    heights = df["ymax"].astype(float) - df["ymin"].astype(float)
    if not np.allclose(widths.to_numpy(), 80.0):
        raise ValueError("Not all cells have 80 m width")
    if not np.allclose(heights.to_numpy(), 80.0):
        raise ValueError("Not all cells have 80 m height")

    df["centroid_x"] = (df["xmin"].astype(float) + df["xmax"].astype(float)) / 2.0
    df["centroid_y"] = (df["ymin"].astype(float) + df["ymax"].astype(float)) / 2.0
    return df


def load_label_mask(split: str):
    y = np.load(LABEL_PATCHES[split], allow_pickle=False).astype(np.int8)
    mask = np.load(VALID_MASKS[split], allow_pickle=False).astype(bool)
    expected = (EXPECTED_PATCHES[split], 16, 16)
    if y.shape != expected or mask.shape != expected:
        raise ValueError(f"{split} label/mask shape mismatch")
    if int(mask.sum()) != EXPECTED_VALID[split]:
        raise ValueError(f"{split} valid mask count mismatch")
    return y, mask


def crosscheck_mapping(mapping_split, y_patch, valid_mask, split):
    p = mapping_split["patch_order"].to_numpy(dtype=int)
    r = mapping_split["cell_row"].to_numpy(dtype=int)
    c = mapping_split["cell_col"].to_numpy(dtype=int)

    if not np.all(valid_mask[p, r, c]):
        raise ValueError(f"{split}: mapping contains invalid-mask cells")
    if not np.array_equal(
        y_patch[p, r, c].astype(np.int8),
        mapping_split["label"].to_numpy(dtype=np.int8),
    ):
        raise ValueError(f"{split}: mapping labels != label array")


def infer_train_s2_grid():
    x = np.load(X_TRAIN, allow_pickle=False)
    expected = (EXPECTED_PATCHES["train"], 128, 128, 4)
    if x.shape != expected or x.dtype != np.float32:
        raise ValueError(f"Unexpected Train S2 input: {x.shape}, {x.dtype}")
    if not np.isfinite(x).all():
        raise ValueError("Train S2 contains non-finite values")

    model = tf.keras.models.load_model(S2_MODEL, compile=False)
    probs = model.predict(x, batch_size=16, verbose=0).astype(np.float32)
    del x

    expected_p = (EXPECTED_PATCHES["train"], 16, 16, 1)
    if probs.shape != expected_p:
        raise ValueError(f"Unexpected Train S2 prediction shape: {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("Train S2 probabilities non-finite")
    return probs[..., 0]


def load_frozen_val_s2_grid():
    probs = np.load(S2_VAL_PROBS_FROZEN, allow_pickle=False).astype(np.float32)
    if probs.shape == (EXPECTED_PATCHES["val"], 16, 16, 1):
        probs = probs[..., 0]
    expected = (EXPECTED_PATCHES["val"], 16, 16)
    if probs.shape != expected:
        raise ValueError(f"Frozen Val S2 prediction shape: {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("Frozen Val S2 probabilities non-finite")
    return probs


def load_train_val_fusion():
    df = pd.read_csv(STEP20A)
    required = {
        "split", "global_cell_row", "global_cell_col",
        "slot_count", "fusion_probability",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Step20A missing columns: {sorted(missing)}")

    df = df.loc[df["split"].isin(["train", "val"])].copy()
    for split in ["train", "val"]:
        if int((df["split"] == split).sum()) != EXPECTED_COVERED[split]:
            raise ValueError(f"Step20A {split} covered count mismatch")
    if df.duplicated(["global_cell_row", "global_cell_col"]).any():
        raise ValueError("Duplicate Train/Val covered cells in Step20A")
    if not np.isfinite(df["fusion_probability"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite Train/Val Fusion probabilities")

    return df[
        ["split", "global_cell_row", "global_cell_col",
         "slot_count", "fusion_probability"]
    ].copy()


def make_train_val_rows(split, mapping, s2_grid, fusion_tv):
    m = (
        mapping.loc[mapping["split"] == split]
        .copy()
        .sort_values(["patch_order", "cell_row", "cell_col"], kind="stable")
        .reset_index(drop=True)
    )

    p = m["patch_order"].to_numpy(dtype=int)
    r = m["cell_row"].to_numpy(dtype=int)
    c = m["cell_col"].to_numpy(dtype=int)

    m["s2_probability"] = s2_grid[p, r, c].astype(np.float64)
    m["s2_prediction"] = (
        m["s2_probability"].to_numpy(dtype=float) >= S2_THRESHOLD
    ).astype(np.int8)

    cov = fusion_tv.loc[
        fusion_tv["split"] == split,
        ["global_cell_row", "global_cell_col", "slot_count", "fusion_probability"],
    ].copy()

    m = m.merge(
        cov,
        on=["global_cell_row", "global_cell_col"],
        how="left",
        validate="one_to_one",
    )
    m["has_gsv"] = m["fusion_probability"].notna()

    if int(m["has_gsv"].sum()) != EXPECTED_COVERED[split]:
        raise ValueError(f"{split}: covered count mismatch")
    if int((~m["has_gsv"]).sum()) != EXPECTED_NO_GSV[split]:
        raise ValueError(f"{split}: no-GSV count mismatch")

    m["slot_count"] = m["slot_count"].fillna(0).astype(int)

    fusion_pred = pd.Series(pd.NA, index=m.index, dtype="Int64")
    covered_mask = m["has_gsv"].to_numpy(dtype=bool)
    fusion_pred.loc[covered_mask] = (
        m.loc[covered_mask, "fusion_probability"].to_numpy(dtype=float)
        >= FUSION_THRESHOLD
    ).astype(np.int8)
    m["fusion_prediction"] = fusion_pred

    final_pred = m["s2_prediction"].to_numpy(dtype=np.int8).copy()
    final_pred[covered_mask] = (
        m.loc[covered_mask, "fusion_prediction"]
        .astype(int)
        .to_numpy(dtype=np.int8)
    )
    m["final_prediction"] = final_pred
    m["prediction_source"] = np.where(covered_mask, "fusion_mean", "s2_only")

    if split == "train":
        m["s2_probability_source"] = "frozen_model_posthoc_train_forward_pass"
    else:
        m["s2_probability_source"] = "frozen_original_validation_probability_artifact"

    m["fusion_probability_source"] = np.where(
        covered_mask,
        "step20a_frozen_model_forward_pass",
        "not_applicable_no_gsv",
    )
    m["final_prediction_origin"] = "posthoc_full_area_mapping_assembly"
    m["evidence_role"] = "model-development descriptive mapping output"
    return m


def make_test_rows(mapping):
    m = (
        mapping.loc[mapping["split"] == "test"]
        .copy()
        .sort_values(["patch_order", "cell_row", "cell_col"], kind="stable")
        .reset_index(drop=True)
    )

    frozen_all = pd.read_csv(TEST_ALL_VALID)
    required = {
        "patch_order", "cell_row", "cell_col", "label",
        "s2_probability", "s2_prediction", "conditional_prediction",
    }
    missing = required - set(frozen_all.columns)
    if missing:
        raise ValueError(f"Frozen Test all-valid missing: {sorted(missing)}")
    if len(frozen_all) != EXPECTED_VALID["test"]:
        raise ValueError("Frozen Test all-valid count mismatch")

    keys = ["patch_order", "cell_row", "cell_col"]
    if frozen_all.duplicated(keys).any():
        raise ValueError("Duplicate frozen Test all-valid addresses")

    m = m.merge(
        frozen_all,
        on=keys,
        how="left",
        suffixes=("_mapping", "_frozen"),
        validate="one_to_one",
    )

    if not np.array_equal(
        m["label_mapping"].to_numpy(dtype=np.int8),
        m["label_frozen"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("Frozen Step15 Test labels != mapping labels")

    m["label"] = m["label_mapping"].astype(np.int8)
    m.drop(columns=["label_mapping", "label_frozen"], inplace=True)

    covered = pd.read_csv(TEST_COVERED)
    required_cov = {
        "patch_order", "cell_row", "cell_col", "slot_count",
        "fusion_mean_probability", "fusion_mean_prediction",
    }
    missing_cov = required_cov - set(covered.columns)
    if missing_cov:
        raise ValueError(f"Frozen Test covered missing: {sorted(missing_cov)}")
    if len(covered) != EXPECTED_COVERED["test"]:
        raise ValueError("Frozen Test covered count mismatch")
    if covered.duplicated(keys).any():
        raise ValueError("Duplicate frozen Test covered addresses")

    covered = covered[
        keys + [
            "slot_count",
            "fusion_mean_probability",
            "fusion_mean_prediction",
        ]
    ].rename(
        columns={
            "fusion_mean_probability": "fusion_probability",
            "fusion_mean_prediction": "fusion_prediction",
        }
    )

    m = m.merge(covered, on=keys, how="left", validate="one_to_one")
    m["has_gsv"] = m["fusion_probability"].notna()
    m["slot_count"] = m["slot_count"].fillna(0).astype(int)

    if int(m["has_gsv"].sum()) != EXPECTED_COVERED["test"]:
        raise ValueError("Frozen Test covered count mismatch after join")
    if int((~m["has_gsv"]).sum()) != EXPECTED_NO_GSV["test"]:
        raise ValueError("Frozen Test no-GSV count mismatch after join")

    m["final_prediction"] = m["conditional_prediction"].astype(np.int8)
    m.drop(columns=["conditional_prediction"], inplace=True)

    covered_mask = m["has_gsv"].to_numpy(dtype=bool)
    if not np.array_equal(
        m.loc[covered_mask, "final_prediction"].to_numpy(dtype=np.int8),
        m.loc[covered_mask, "fusion_prediction"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("Frozen Test routing mismatch on covered cells")
    if not np.array_equal(
        m.loc[~covered_mask, "final_prediction"].to_numpy(dtype=np.int8),
        m.loc[~covered_mask, "s2_prediction"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("Frozen Test routing mismatch on no-GSV cells")

    m["prediction_source"] = np.where(covered_mask, "fusion_mean", "s2_only")
    m["s2_probability_source"] = "frozen_step15_final_test"
    m["fusion_probability_source"] = np.where(
        covered_mask, "frozen_step15_covered_test", "not_applicable_no_gsv"
    )
    m["final_prediction_origin"] = "frozen_step15_final_test"
    m["evidence_role"] = "independent held-out mapping output"
    return m


def add_change_fields(df):
    s2 = df["s2_prediction"].to_numpy(dtype=np.int8)
    final = df["final_prediction"].to_numpy(dtype=np.int8)

    df["s2_to_final_changed"] = s2 != final

    change = np.empty(len(df), dtype=object)
    change[(s2 == 0) & (final == 0)] = "unchanged_non_informal"
    change[(s2 == 1) & (final == 1)] = "unchanged_informal"
    change[(s2 == 0) & (final == 1)] = "changed_0_to_1"
    change[(s2 == 1) & (final == 0)] = "changed_1_to_0"
    df["s2_to_final_change"] = change

    df["heldout_test_transition"] = pd.Series(pd.NA, index=df.index, dtype="string")

    idx = df.index[df["split"].eq("test")].to_numpy()
    y = df.loc[idx, "label"].to_numpy(dtype=np.int8)
    s2_t = df.loc[idx, "s2_prediction"].to_numpy(dtype=np.int8)
    final_t = df.loc[idx, "final_prediction"].to_numpy(dtype=np.int8)

    s2_correct = s2_t == y
    final_correct = final_t == y
    trans = np.empty(len(idx), dtype=object)

    trans[s2_correct & final_correct] = "stable_correct"
    trans[(~s2_correct) & (~final_correct)] = "stable_wrong"
    trans[(~s2_correct) & final_correct & (y == 0)] = "fp_to_correct"
    trans[(~s2_correct) & final_correct & (y == 1)] = "fn_to_correct"
    trans[s2_correct & (~final_correct) & (y == 0)] = "correct_to_fp"
    trans[s2_correct & (~final_correct) & (y == 1)] = "correct_to_fn"

    if pd.isna(trans).any():
        raise ValueError("Unclassified held-out Test transition")
    df.loc[idx, "heldout_test_transition"] = trans
    return df


def normalize_columns(df):
    ordered = [
        "split", "patch_order", "original_patch_index", "row_index_in_split",
        "cell_row", "cell_col", "global_cell_row", "global_cell_col",
        "reference_label", "label", "informal_fraction",
        "valid_10m_pixel_count", "valid_coverage_fraction",
        "has_gsv", "slot_count",
        "s2_probability", "s2_prediction",
        "fusion_probability", "fusion_prediction",
        "final_prediction", "prediction_source",
        "s2_to_final_changed", "s2_to_final_change",
        "heldout_test_transition",
        "s2_probability_source", "fusion_probability_source",
        "final_prediction_origin", "evidence_role",
        "xmin", "ymin", "xmax", "ymax", "centroid_x", "centroid_y",
    ]
    missing = [c for c in ordered if c not in df.columns]
    if missing:
        raise ValueError(f"Final table missing columns: {missing}")
    return df[ordered].copy()


def build_summary(full):
    rows = []
    for split in ["train", "val", "test", "all"]:
        d = full if split == "all" else full.loc[full["split"] == split]
        n = len(d)
        n_cov = int(d["has_gsv"].sum())
        n_changed = int(d["s2_to_final_changed"].sum())

        rows.append({
            "split": split,
            "n_valid_cells": n,
            "n_gsv_covered": n_cov,
            "n_no_gsv": n - n_cov,
            "reference_positive": int((d["label"] == 1).sum()),
            "reference_negative": int((d["label"] == 0).sum()),
            "s2_pred_positive": int((d["s2_prediction"] == 1).sum()),
            "s2_pred_negative": int((d["s2_prediction"] == 0).sum()),
            "final_pred_positive": int((d["final_prediction"] == 1).sum()),
            "final_pred_negative": int((d["final_prediction"] == 0).sum()),
            "n_s2_to_final_changed": n_changed,
            "changed_percent_of_all_valid": 100.0 * n_changed / n,
            "changed_percent_of_gsv_covered": (
                100.0 * n_changed / n_cov if n_cov else np.nan
            ),
            "n_changed_0_to_1": int(
                (d["s2_to_final_change"] == "changed_0_to_1").sum()
            ),
            "n_changed_1_to_0": int(
                (d["s2_to_final_change"] == "changed_1_to_0").sum()
            ),
        })
    return pd.DataFrame(rows)


def build_test_transition_summary(full):
    test = full.loc[full["split"] == "test"]
    counts = test["heldout_test_transition"].value_counts().to_dict()
    states = [
        "stable_correct", "stable_wrong",
        "fp_to_correct", "fn_to_correct",
        "correct_to_fp", "correct_to_fn",
    ]
    return pd.DataFrame([
        {
            "heldout_test_transition": s,
            "n_cells": int(counts.get(s, 0)),
            "percent_of_all_valid_test": 100.0 * int(counts.get(s, 0)) / EXPECTED_VALID["test"],
        }
        for s in states
    ])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    protected = [OUT_CELLWISE, OUT_SUMMARY, OUT_TEST_TRANSITIONS, OUT_AUDIT_JSON]
    existing = [p for p in protected if p.exists()]
    if existing:
        raise FileExistsError(
            "Protected Step21A outputs already exist; refusing overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )

    for p in [
        S2_MODEL, X_TRAIN, S2_VAL_PROBS_FROZEN, CELL_MAPPING, STEP20A,
        TEST_ALL_VALID, TEST_COVERED,
        *LABEL_PATCHES.values(), *VALID_MASKS.values(),
    ]:
        require_file(p)

    print("=" * 100)
    print("Nairobi v2 — Step 21A Final Full-Area Mapping Dataset")
    print("=" * 100)
    print("Training: NO")
    print("Threshold tuning: NO")
    print("Model selection: NO")
    print("Model modification: NO")
    print("Test prediction recomputation: NO")
    print("Full-area Fusion pseudo-prediction for no-GSV cells: NO")
    print("Mixed conditional probability constructed: NO")
    print("Spatial aggregation/smoothing/interpolation: NO")
    print("Routing: covered -> Fusion Mean @ 0.59; no-GSV -> S2-only @ 0.78")
    print()

    mapping = load_mapping()
    train_y, train_mask = load_label_mask("train")
    val_y, val_mask = load_label_mask("val")
    crosscheck_mapping(mapping.loc[mapping["split"] == "train"], train_y, train_mask, "train")
    crosscheck_mapping(mapping.loc[mapping["split"] == "val"], val_y, val_mask, "val")
    print("[1/7] 112,600-cell spatial mapping + label alignment: PASSED")

    train_s2_grid = infer_train_s2_grid()
    val_s2_grid = load_frozen_val_s2_grid()
    print("[2/7] S2 sources: Train frozen-model inference + frozen Val artifact: PASSED")

    fusion_tv = load_train_val_fusion()
    print("[3/7] Train / Val covered-cell Fusion probabilities from Step20A: PASSED")

    train_rows = make_train_val_rows("train", mapping, train_s2_grid, fusion_tv)
    val_rows = make_train_val_rows("val", mapping, val_s2_grid, fusion_tv)
    print("[4/7] Train / Val frozen conditional-routing assembly: PASSED")

    test_rows = make_test_rows(mapping)
    print("[5/7] Frozen Step15 Test predictions imported unchanged: PASSED")

    full = pd.concat([train_rows, val_rows, test_rows], ignore_index=True)
    if len(full) != EXPECTED_VALID["all"]:
        raise ValueError("Final full-area row count mismatch")
    if full.duplicated(["global_cell_row", "global_cell_col"]).any():
        raise ValueError("Duplicate global cells in final dataset")

    full = add_change_fields(full)
    full = normalize_columns(full)

    if int(full["has_gsv"].sum()) != EXPECTED_COVERED["all"]:
        raise ValueError("Final covered-cell count mismatch")
    if int((~full["has_gsv"]).sum()) != EXPECTED_NO_GSV["all"]:
        raise ValueError("Final no-GSV count mismatch")

    no_gsv_changed = int(
        full.loc[~full["has_gsv"], "s2_to_final_changed"].sum()
    )
    if no_gsv_changed != 0:
        raise ValueError(f"No-GSV cells changed relative to S2: {no_gsv_changed}")

    forbidden = {"conditional_probability", "final_probability", "routed_probability"}
    if forbidden & set(full.columns):
        raise ValueError("Forbidden mixed probability field present")

    print("[6/7] Routing/change/Test-transition hard guards: PASSED")

    full.to_csv(OUT_CELLWISE, index=False, quoting=csv.QUOTE_MINIMAL)
    summary = build_summary(full)
    summary.to_csv(OUT_SUMMARY, index=False)
    test_summary = build_test_transition_summary(full)
    test_summary.to_csv(OUT_TEST_TRANSITIONS, index=False)
    print("[7/7] Final dataset + summaries written: PASSED")

    audit = {
        "status": "PASSED",
        "step": "21a",
        "purpose": (
            "Build one reusable 112,600-cell dataset supporting Reference, "
            "S2-only, Final conditional-routing, S2-to-Final hard-class change, "
            "and held-out Test transition maps."
        ),
        "training_performed": False,
        "threshold_tuning_performed": False,
        "model_selection_performed": False,
        "model_modified": False,
        "test_predictions_recomputed": False,
        "full_area_fusion_for_no_gsv_constructed": False,
        "mixed_conditional_probability_constructed": False,
        "spatial_aggregation_smoothing_or_interpolation": False,
        "routing": {
            "gsv_covered": {"model": "Fusion Mean", "threshold": FUSION_THRESHOLD},
            "gsv_unavailable": {"model": "S2-only", "threshold": S2_THRESHOLD},
        },
        "counts": {
            "valid": EXPECTED_VALID,
            "covered": EXPECTED_COVERED,
            "no_gsv": EXPECTED_NO_GSV,
        },
        "s2_sources": {
            "train": "Frozen S2 model forward pass for post-hoc mapping only.",
            "val": str(S2_VAL_PROBS_FROZEN),
            "test": str(TEST_ALL_VALID),
        },
        "fusion_sources": {
            "train_val_covered": str(STEP20A),
            "test_covered": str(TEST_COVERED),
            "no_gsv": "not applicable; routed to S2-only",
        },
        "evidence_roles": {
            "train": "model-development descriptive mapping output",
            "val": "model-development descriptive mapping output",
            "test": "independent held-out mapping output",
        },
        "heldout_test_transition_definition": (
            "Correctness-transition labels are populated only for Test."
        ),
        "crs_statement": (
            "Coordinates are inherited projected metric coordinates; "
            "no EPSG code is asserted here."
        ),
        "outputs": {
            "cellwise_csv": str(OUT_CELLWISE),
            "summary_csv": str(OUT_SUMMARY),
            "test_transition_summary_csv": str(OUT_TEST_TRANSITIONS),
        },
    }
    OUT_AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    OUT_AUDIT_TXT.write_text(
        "\n".join([
            "=" * 100,
            "Nairobi v2 — Step 21A Final Full-Area Mapping Dataset Audit",
            "=" * 100,
            "STATUS: PASSED",
            "",
            "Training performed: NO",
            "Threshold tuning performed: NO",
            "Model selection performed: NO",
            "Model modification: NO",
            "Test predictions recomputed: NO",
            "Full-area Fusion pseudo-predictions for no-GSV cells: NO",
            "Mixed conditional probability constructed: NO",
            "Spatial aggregation/smoothing/interpolation: NO",
            "",
            "Frozen routing:",
            "  GSV-covered -> Fusion Mean @ 0.59",
            "  no-GSV      -> S2-only @ 0.78",
            "",
            summary.to_csv(index=False).strip(),
            "",
            "Held-out Test transitions:",
            test_summary.to_csv(index=False).strip(),
            "",
            f"Cellwise: {OUT_CELLWISE}",
            f"Summary: {OUT_SUMMARY}",
            f"Test transitions: {OUT_TEST_TRANSITIONS}",
        ]),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("STEP 21A STATUS: PASSED")
    print("=" * 100)
    print(summary.to_string(index=False))
    print()
    print("Held-out Test transitions:")
    print(test_summary.to_string(index=False))
    print()
    print("Cellwise:", OUT_CELLWISE)
    print("Summary:", OUT_SUMMARY)
    print("Test transitions:", OUT_TEST_TRANSITIONS)
    print("Audit:", OUT_AUDIT_JSON)


if __name__ == "__main__":
    main()
