#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np
import tensorflow as tf


PROJECT = Path.home() / "projects" / "nairobi_v2"
SEED = 42
N_COVERED = 5744
FEATURE_DIM = 256
EPS = 1e-7

# Frozen artifacts
FREEZE_MANIFEST = (
    PROJECT / "results/final_protocol_v2/seed_42/final_protocol_manifest_v2.json"
)
FINAL_TEST_AUDIT = (
    PROJECT / "results/final_test_v2/seed_42/final_test_audit_v2.json"
)
COVERED_PREDICTIONS = (
    PROJECT / "results/final_test_v2/seed_42/covered_test_predictions_v2.csv"
)
COVERED_CELL_INDEX = (
    PROJECT / "results/final_test_v2/seed_42/covered_test_cell_index_v2.csv"
)
X_TEST_S2 = PROJECT / "scratch/features/s2_inputs/X_test_s2_v2.npy"

S2_ENCODER = PROJECT / "models/s2_only_v2/seed_42/s2_encoder_best_v2.keras"
FUSION_MODEL = PROJECT / "models/fusion_mean_pool_v2/seed_42/fusion_mean_pool_best_v2.keras"
FUSION_SCALER = (
    PROJECT
    / "results/fusion_mean_pool_v2/seed_42/"
      "fusion_mean_pool_standardization_train_only_v2.npz"
)

OUT_DIR = PROJECT / "results/xai_modality_v2/seed_42"
CELLWISE_CSV = OUT_DIR / "xai_modality_delta_cellwise_v2.csv"
GROUP_SUMMARY_CSV = OUT_DIR / "xai_corrected_degraded_summary_v2.csv"
AUDIT_JSON = OUT_DIR / "xai_modality_audit_v2.json"
AUDIT_TXT = OUT_DIR / "xai_modality_audit_v2.txt"
NEUTRAL_PROBS_NPY = OUT_DIR / "fusion_gsv_neutralized_probabilities_v2.npy"
DELTA_LOGIT_NPY = OUT_DIR / "delta_gsv_logit_v2.npy"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_clean_start() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protected = [
        CELLWISE_CSV,
        GROUP_SUMMARY_CSV,
        AUDIT_JSON,
        AUDIT_TXT,
        NEUTRAL_PROBS_NPY,
        DELTA_LOGIT_NPY,
    ]
    existing = [p for p in protected if p.exists()]
    if existing:
        raise FileExistsError(
            "XAI Step 1-2 outputs already exist; refusing overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )


def parse_int(row: dict, key: str) -> int:
    return int(float(str(row[key]).strip()))


def parse_float(row: dict, key: str) -> float:
    return float(str(row[key]).strip())


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def verify_frozen_artifacts() -> dict:
    protocol = load_json(FREEZE_MANIFEST)
    if protocol.get("status") != "FROZEN":
        raise ValueError("Final protocol is not FROZEN")
    if int(protocol.get("primary_seed", -1)) != SEED:
        raise ValueError("Primary seed is not 42")

    test_audit = load_json(FINAL_TEST_AUDIT)
    if test_audit.get("status") != "PASSED":
        raise ValueError("Final test audit is not PASSED")
    if int(test_audit.get("primary_seed", -1)) != SEED:
        raise ValueError("Final test seed is not 42")
    if test_audit.get("training_performed") is not False:
        raise ValueError("Final test audit unexpectedly reports training")
    if test_audit.get("threshold_selection_performed") is not False:
        raise ValueError("Final test audit unexpectedly reports threshold selection")
    if test_audit.get("model_selection_performed") is not False:
        raise ValueError("Final test audit unexpectedly reports model selection")

    frozen_hashes = protocol["frozen_artifact_hashes"]
    path_map = {
        "s2_encoder": S2_ENCODER,
        "fusion_mean_model": FUSION_MODEL,
        "fusion_train_only_scaler": FUSION_SCALER,
    }
    for name, path in path_map.items():
        rec = frozen_hashes[name]
        if Path(rec["path"]) != path:
            raise ValueError(f"Frozen path mismatch: {name}")
        actual = sha256_file(path)
        if actual != rec["sha256"]:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")

    return test_audit


def load_and_align_rows():
    pred_rows = read_csv_rows(COVERED_PREDICTIONS)
    idx_rows = read_csv_rows(COVERED_CELL_INDEX)

    if len(pred_rows) != N_COVERED or len(idx_rows) != N_COVERED:
        raise ValueError(
            f"Covered-row counts are not {N_COVERED}: "
            f"{len(pred_rows)}, {len(idx_rows)}"
        )

    pred_by_order = {parse_int(r, "covered_order"): r for r in pred_rows}
    idx_by_order = {parse_int(r, "covered_order"): r for r in idx_rows}

    expected = set(range(N_COVERED))
    if set(pred_by_order) != expected or set(idx_by_order) != expected:
        raise ValueError("covered_order is not exact 0..5743")

    aligned = []
    identity_fields = [
        "patch_order",
        "cell_row",
        "cell_col",
        "global_cell_row",
        "global_cell_col",
        "label",
        "slot_count",
    ]

    for order in range(N_COVERED):
        p = pred_by_order[order]
        i = idx_by_order[order]
        for field in identity_fields:
            if parse_int(p, field) != parse_int(i, field):
                raise ValueError(
                    f"Prediction/index identity mismatch at order={order}, field={field}"
                )
        aligned.append((p, i))

    return aligned


def extract_s2_features(aligned):
    x_test = np.load(X_TEST_S2, allow_pickle=False)
    if x_test.shape != (102, 128, 128, 4):
        raise ValueError(f"Unexpected X_test shape: {x_test.shape}")
    if x_test.dtype != np.float32 or not np.isfinite(x_test).all():
        raise ValueError("Invalid S2 test input")

    encoder = tf.keras.models.load_model(S2_ENCODER, compile=False)
    grid = encoder.predict(x_test, batch_size=16, verbose=0).astype(np.float32)
    if grid.shape != (102, 16, 16, FEATURE_DIM):
        raise ValueError(f"Unexpected S2 encoder output: {grid.shape}")

    features = np.zeros((N_COVERED, FEATURE_DIM), dtype=np.float32)
    for order, (_, idx) in enumerate(aligned):
        p = parse_int(idx, "patch_order")
        r = parse_int(idx, "cell_row")
        c = parse_int(idx, "cell_col")
        features[order] = grid[p, r, c]

    if not np.isfinite(features).all():
        raise ValueError("Non-finite covered S2 features")

    return features


def classify_change(y: int, s2_pred: int, fusion_pred: int) -> tuple[str, str]:
    s2_correct = s2_pred == y
    fusion_correct = fusion_pred == y

    if s2_correct and fusion_correct:
        return "both_correct", "both_correct"
    if (not s2_correct) and fusion_correct:
        if y == 0 and s2_pred == 1 and fusion_pred == 0:
            return "s2_wrong_fusion_correct", "s2_fp_to_correct"
        if y == 1 and s2_pred == 0 and fusion_pred == 1:
            return "s2_wrong_fusion_correct", "s2_fn_to_correct"
        raise ValueError("Unexpected corrected prediction pattern")
    if s2_correct and (not fusion_correct):
        if y == 0 and s2_pred == 0 and fusion_pred == 1:
            return "s2_correct_fusion_wrong", "s2_correct_to_fusion_fp"
        if y == 1 and s2_pred == 1 and fusion_pred == 0:
            return "s2_correct_fusion_wrong", "s2_correct_to_fusion_fn"
        raise ValueError("Unexpected degraded prediction pattern")
    return "both_wrong", "both_wrong"


def summarize_group(name: str, mask: np.ndarray, delta: np.ndarray, y: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {
            "group": name,
            "n": 0,
            "percent_of_covered": 0.0,
            "informal_n": 0,
            "formal_noninformal_n": 0,
            "delta_gsv_mean": "",
            "delta_gsv_median": "",
            "delta_gsv_abs_mean": "",
            "delta_gsv_abs_median": "",
            "delta_gsv_positive_percent": "",
            "delta_gsv_negative_percent": "",
        }

    d = delta[mask]
    yy = y[mask]
    return {
        "group": name,
        "n": n,
        "percent_of_covered": n / N_COVERED * 100.0,
        "informal_n": int(np.sum(yy == 1)),
        "formal_noninformal_n": int(np.sum(yy == 0)),
        "delta_gsv_mean": float(np.mean(d)),
        "delta_gsv_median": float(np.median(d)),
        "delta_gsv_abs_mean": float(np.mean(np.abs(d))),
        "delta_gsv_abs_median": float(np.median(np.abs(d))),
        "delta_gsv_positive_percent": float(np.mean(d > 0) * 100.0),
        "delta_gsv_negative_percent": float(np.mean(d < 0) * 100.0),
    }


def main():
    ensure_clean_start()

    print("=" * 92)
    print("Nairobi v2 — XAI Step 1-2: GSV modality influence + corrected/degraded cases")
    print("=" * 92)
    print("Seed: 42")
    print("Training: NO")
    print("Threshold tuning: NO")
    print("Model modification: NO")
    print("Population: frozen 5,744 GSV-covered final-test cells")
    print()

    test_audit = verify_frozen_artifacts()
    print("[1] Frozen protocol/final-test audit/artifact hashes: PASSED")

    aligned = load_and_align_rows()
    print("[2] Covered prediction/index alignment: PASSED")

    s2_features = extract_s2_features(aligned)

    scaler = np.load(FUSION_SCALER, allow_pickle=False)
    required = {"s2_mean", "s2_std", "gsv_mean", "gsv_std"}
    if set(scaler.files) != required:
        raise ValueError(f"Unexpected scaler keys: {set(scaler.files)}")

    s2_mean = scaler["s2_mean"].astype(np.float32)
    s2_std = scaler["s2_std"].astype(np.float32)
    scaler.close()

    if s2_mean.shape != (FEATURE_DIM,) or s2_std.shape != (FEATURE_DIM,):
        raise ValueError("Invalid S2 scaler shape")
    if not np.isfinite(s2_mean).all() or not np.isfinite(s2_std).all():
        raise ValueError("Non-finite S2 scaler")

    s2_z = ((s2_features - s2_mean) / s2_std).astype(np.float32)
    if not np.isfinite(s2_z).all():
        raise ValueError("Non-finite standardized S2 features")

    # GSV-neutralized baseline:
    # standardized GSV = 0 exactly corresponds to the TRAIN mean GSV representation.
    gsv_neutral_z = np.zeros((N_COVERED, FEATURE_DIM), dtype=np.float32)

    fusion_model = tf.keras.models.load_model(FUSION_MODEL, compile=False)
    neutral_probs = fusion_model.predict(
        {
            "s2_feature_256": s2_z,
            "gsv_feature_256": gsv_neutral_z,
        },
        batch_size=128,
        verbose=0,
    ).astype(np.float32)

    if neutral_probs.shape != (N_COVERED, 1):
        raise ValueError(f"Unexpected neutral probability shape: {neutral_probs.shape}")
    neutral_probs = neutral_probs[:, 0]
    if not np.isfinite(neutral_probs).all():
        raise ValueError("Non-finite GSV-neutralized probabilities")

    print("[3] GSV-neutralized Fusion inference: PASSED")
    print("    Baseline = standardized GSV vector of zero = TRAIN-mean GSV representation")

    y = np.zeros(N_COVERED, dtype=np.int8)
    s2_prob = np.zeros(N_COVERED, dtype=np.float64)
    fusion_prob = np.zeros(N_COVERED, dtype=np.float64)
    s2_pred = np.zeros(N_COVERED, dtype=np.int8)
    fusion_pred = np.zeros(N_COVERED, dtype=np.int8)

    group_names = []
    subtype_names = []
    cellwise_rows = []

    for order, (pred, idx) in enumerate(aligned):
        y[order] = parse_int(pred, "label")
        s2_prob[order] = parse_float(pred, "s2_probability")
        fusion_prob[order] = parse_float(pred, "fusion_mean_probability")
        s2_pred[order] = parse_int(pred, "s2_prediction")
        fusion_pred[order] = parse_int(pred, "fusion_mean_prediction")

        group, subtype = classify_change(
            int(y[order]),
            int(s2_pred[order]),
            int(fusion_pred[order]),
        )
        group_names.append(group)
        subtype_names.append(subtype)

    full_logit = logit(fusion_prob)
    neutral_logit = logit(neutral_probs)
    delta = full_logit - neutral_logit

    if not np.isfinite(delta).all():
        raise ValueError("Non-finite ΔGSV")

    for order, (pred, idx) in enumerate(aligned):
        cellwise_rows.append(
            {
                "covered_order": order,
                "patch_order": parse_int(idx, "patch_order"),
                "cell_row": parse_int(idx, "cell_row"),
                "cell_col": parse_int(idx, "cell_col"),
                "global_cell_row": parse_int(idx, "global_cell_row"),
                "global_cell_col": parse_int(idx, "global_cell_col"),
                "label": int(y[order]),
                "slot_count": parse_int(idx, "slot_count"),
                "s2_probability": float(s2_prob[order]),
                "s2_prediction": int(s2_pred[order]),
                "fusion_probability": float(fusion_prob[order]),
                "fusion_prediction": int(fusion_pred[order]),
                "gsv_neutralized_probability": float(neutral_probs[order]),
                "full_fusion_logit": float(full_logit[order]),
                "gsv_neutralized_logit": float(neutral_logit[order]),
                "delta_gsv_logit": float(delta[order]),
                "abs_delta_gsv_logit": float(abs(delta[order])),
                "prediction_change_group": group_names[order],
                "prediction_change_subtype": subtype_names[order],
            }
        )

    group_arr = np.asarray(group_names, dtype=object)
    subtype_arr = np.asarray(subtype_names, dtype=object)

    summary_rows = []
    for name in [
        "all_covered",
        "both_correct",
        "s2_wrong_fusion_correct",
        "s2_correct_fusion_wrong",
        "both_wrong",
        "s2_fp_to_correct",
        "s2_fn_to_correct",
        "s2_correct_to_fusion_fp",
        "s2_correct_to_fusion_fn",
    ]:
        if name == "all_covered":
            mask = np.ones(N_COVERED, dtype=bool)
        elif name in {
            "both_correct",
            "s2_wrong_fusion_correct",
            "s2_correct_fusion_wrong",
            "both_wrong",
        }:
            mask = group_arr == name
        else:
            mask = subtype_arr == name

        summary_rows.append(summarize_group(name, mask, delta, y))

    corrected_n = int(np.sum(group_arr == "s2_wrong_fusion_correct"))
    degraded_n = int(np.sum(group_arr == "s2_correct_fusion_wrong"))
    both_correct_n = int(np.sum(group_arr == "both_correct"))
    both_wrong_n = int(np.sum(group_arr == "both_wrong"))

    if corrected_n + degraded_n + both_correct_n + both_wrong_n != N_COVERED:
        raise ValueError("Paired outcome groups do not sum to 5744")

    s2_correct_n = int(np.sum(s2_pred == y))
    fusion_correct_n = int(np.sum(fusion_pred == y))
    if fusion_correct_n - s2_correct_n != corrected_n - degraded_n:
        raise ValueError("Paired correctness accounting identity failed")

    # Commit outputs only after all checks have passed.
    np.save(NEUTRAL_PROBS_NPY, neutral_probs.astype(np.float32), allow_pickle=False)
    np.save(DELTA_LOGIT_NPY, delta.astype(np.float32), allow_pickle=False)
    write_csv(CELLWISE_CSV, cellwise_rows)
    write_csv(GROUP_SUMMARY_CSV, summary_rows)

    audit = {
        "status": "PASSED",
        "experiment": "XAI Step 1-2: GSV modality influence and corrected/degraded analysis",
        "seed": SEED,
        "population": "Frozen 5,744 GSV-covered final-test cells",
        "training_performed": False,
        "threshold_tuning_performed": False,
        "model_modified": False,
        "attribution_definition": (
            "delta_gsv_logit = logit(full Fusion probability) - "
            "logit(Fusion probability with standardized GSV vector set to zero)"
        ),
        "neutral_baseline_interpretation": (
            "Standardized GSV=0 corresponds to the training-set mean GSV representation. "
            "This is an attribution baseline, not a real missing-GSV sample and not a causal effect."
        ),
        "paired_outcomes": {
            "both_correct": both_correct_n,
            "s2_wrong_fusion_correct": corrected_n,
            "s2_correct_fusion_wrong": degraded_n,
            "both_wrong": both_wrong_n,
            "net_correct_cells_fusion_minus_s2": corrected_n - degraded_n,
            "s2_correct_cells": s2_correct_n,
            "fusion_correct_cells": fusion_correct_n,
        },
        "delta_gsv": {
            "mean": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "mean_absolute": float(np.mean(np.abs(delta))),
            "median_absolute": float(np.median(np.abs(delta))),
            "positive_percent": float(np.mean(delta > 0) * 100.0),
            "negative_percent": float(np.mean(delta < 0) * 100.0),
        },
        "outputs": {
            "cellwise_csv": str(CELLWISE_CSV),
            "group_summary_csv": str(GROUP_SUMMARY_CSV),
            "neutralized_probabilities_npy": str(NEUTRAL_PROBS_NPY),
            "delta_logit_npy": str(DELTA_LOGIT_NPY),
        },
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    summary_by_name = {r["group"]: r for r in summary_rows}

    lines = [
        "=" * 92,
        "Nairobi v2 — XAI Step 1-2 Audit",
        "=" * 92,
        "STATUS: PASSED",
        "Seed: 42",
        "Population: 5,744 frozen GSV-covered final-test cells",
        "Training performed: NO",
        "Threshold tuning performed: NO",
        "Model modification: NO",
        "",
        "GSV modality attribution:",
        "  ΔGSV = logit(full Fusion) - logit(GSV-neutralized Fusion)",
        "  GSV-neutralized = standardized GSV vector 0 = TRAIN-mean GSV representation",
        "  Interpretation: prediction influence/sensitivity, NOT causal effect",
        f"  Mean ΔGSV: {float(np.mean(delta)):.6f}",
        f"  Median ΔGSV: {float(np.median(delta)):.6f}",
        f"  Mean |ΔGSV|: {float(np.mean(np.abs(delta))):.6f}",
        f"  Median |ΔGSV|: {float(np.median(np.abs(delta))):.6f}",
        f"  ΔGSV > 0: {float(np.mean(delta > 0) * 100.0):.2f}%",
        f"  ΔGSV < 0: {float(np.mean(delta < 0) * 100.0):.2f}%",
        "",
        "Paired S2 vs Fusion outcomes:",
        f"  Both correct: {both_correct_n}",
        f"  S2 wrong -> Fusion correct: {corrected_n}",
        f"  S2 correct -> Fusion wrong: {degraded_n}",
        f"  Both wrong: {both_wrong_n}",
        f"  Net correct cells (Fusion - S2): {corrected_n - degraded_n}",
        "",
        "Corrected/degraded subtypes:",
        f"  S2 FP -> correct: {summary_by_name['s2_fp_to_correct']['n']}",
        f"  S2 FN -> correct: {summary_by_name['s2_fn_to_correct']['n']}",
        f"  S2 correct -> Fusion FP: {summary_by_name['s2_correct_to_fusion_fp']['n']}",
        f"  S2 correct -> Fusion FN: {summary_by_name['s2_correct_to_fusion_fn']['n']}",
        "",
        "Outputs:",
        f"  {CELLWISE_CSV}",
        f"  {GROUP_SUMMARY_CSV}",
        f"  {AUDIT_JSON}",
    ]
    AUDIT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
