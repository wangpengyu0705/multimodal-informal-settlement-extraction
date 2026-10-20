#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import tensorflow as tf


# =============================================================================
# Frozen final-test configuration
# =============================================================================

PROJECT = Path.home() / "projects" / "nairobi_v2"
SEED = 42

PROTOCOL_MANIFEST = (
    PROJECT
    / "results/final_protocol_v2/seed_42/final_protocol_manifest_v2.json"
)

S2_MODEL_PATH = (
    PROJECT / "models/s2_only_v2/seed_42/s2_only_model_best_v2.keras"
)
S2_ENCODER_PATH = (
    PROJECT / "models/s2_only_v2/seed_42/s2_encoder_best_v2.keras"
)
GSV_MODEL_PATH = (
    PROJECT / "models/gsv_mean_pool_v2/seed_42/gsv_only_mean_pool_best_v2.keras"
)
GSV_ENCODER_PATH = (
    PROJECT / "models/gsv_mean_pool_v2/seed_42/gsv_encoder_mean_pool_best_v2.keras"
)
FUSION_MODEL_PATH = (
    PROJECT / "models/fusion_mean_pool_v2/seed_42/fusion_mean_pool_best_v2.keras"
)
FUSION_SCALER_PATH = (
    PROJECT
    / "results/fusion_mean_pool_v2/seed_42/"
      "fusion_mean_pool_standardization_train_only_v2.npz"
)

S2_TEST_INPUT = (
    PROJECT / "scratch/features/s2_inputs/X_test_s2_v2.npy"
)
TEST_LABELS = (
    PROJECT / "metadata/latest_aligned_v2/test_label_patches_v2.npy"
)
TEST_VALID_MASK = (
    PROJECT / "metadata/latest_aligned_v2/test_valid_label_mask_v2.npy"
)

GSV_EMBED_PATH = (
    PROJECT
    / "scratch/features/gsv_embeddings/resnet50_imagenet_v2/"
      "gsv_resnet50_embeddings_v2.npy"
)
GSV_EMBED_INDEX = (
    PROJECT
    / "scratch/features/gsv_embeddings/resnet50_imagenet_v2/"
      "gsv_resnet50_embedding_index_v2.csv"
)

OUT_DIR = PROJECT / "results/final_test_v2/seed_42"
COVERED_PRED_CSV = OUT_DIR / "covered_test_predictions_v2.csv"
ALL_VALID_PRED_CSV = OUT_DIR / "all_valid_test_predictions_v2.csv"
METRICS_CSV = OUT_DIR / "final_test_metrics_v2.csv"
AUDIT_JSON = OUT_DIR / "final_test_audit_v2.json"
AUDIT_TXT = OUT_DIR / "final_test_audit_v2.txt"

# Prediction arrays retained for downstream XAI without re-evaluating/reselecting.
S2_TEST_PROBS_GRID = OUT_DIR / "s2_test_probabilities_grid_v2.npy"
GSV_TEST_PROBS_COVERED = OUT_DIR / "gsv_mean_test_probabilities_covered_v2.npy"
FUSION_TEST_PROBS_COVERED = OUT_DIR / "fusion_mean_test_probabilities_covered_v2.npy"
COVERED_CELL_INDEX = OUT_DIR / "covered_test_cell_index_v2.csv"

K = 7
BASE_DIM = 2048
FEATURE_DIM = 256
EXPECTED_IMAGES = 34207
EXPECTED_TEST_IMAGES = 7675
EXPECTED_TEST_PATCHES = 102
EXPECTED_ALL_TEST_POSITIONS = 102 * 16 * 16
EXPECTED_ALL_VALID = 22356
EXPECTED_ALL_NEG = 20827
EXPECTED_ALL_POS = 1529
EXPECTED_COVERED = 5744
EXPECTED_NO_GSV = 16612

S2_THRESHOLD = 0.78
GSV_THRESHOLD = 0.69
FUSION_THRESHOLD = 0.59


# =============================================================================
# Keras custom layer
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="NairobiV2")
class MaskedMeanPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        z, slot_mask = inputs
        mask = tf.cast(slot_mask, z.dtype)[..., None]
        numerator = tf.reduce_sum(z * mask, axis=1)
        denominator = tf.reduce_sum(mask, axis=1)
        return tf.math.divide_no_nan(numerator, denominator)


# =============================================================================
# Utilities
# =============================================================================

def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_int(value, name: str) -> int:
    value = str(value).strip()
    if value == "":
        raise ValueError(f"Empty integer field: {name}")
    return int(float(value))


def choose(fieldnames, candidates, what):
    for name in candidates:
        if name in fieldnames:
            return name
    raise ValueError(f"Missing {what}; columns={fieldnames}")


def ensure_clean_start():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protected = [
        COVERED_PRED_CSV,
        ALL_VALID_PRED_CSV,
        METRICS_CSV,
        AUDIT_JSON,
        AUDIT_TXT,
        S2_TEST_PROBS_GRID,
        GSV_TEST_PROBS_COVERED,
        FUSION_TEST_PROBS_COVERED,
        COVERED_CELL_INDEX,
    ]
    existing = [p for p in protected if p.exists()]
    if existing:
        raise FileExistsError(
            "FINAL TEST outputs already exist. Refusing to overwrite/re-run:\n"
            + "\n".join(str(p) for p in existing)
        )


def verify_frozen_protocol():
    if not PROTOCOL_MANIFEST.is_file():
        raise FileNotFoundError(PROTOCOL_MANIFEST)

    protocol = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))

    if protocol.get("status") != "FROZEN":
        raise ValueError("Protocol status is not FROZEN")
    if int(protocol.get("primary_seed", -1)) != SEED:
        raise ValueError("Protocol primary seed is not 42")
    if protocol.get("test_data_opened_by_this_script") is not False:
        raise ValueError("Freeze manifest does not confirm test remained sealed")
    if protocol.get("test_data_evaluated_by_this_script") is not False:
        raise ValueError("Freeze manifest does not confirm test remained unevaluated")

    models = protocol["final_models"]
    frozen_thresholds = {
        "s2_only": float(models["s2_only"]["threshold"]),
        "gsv_mean": float(models["gsv_mean"]["threshold"]),
        "fusion_mean": float(models["fusion_mean"]["threshold"]),
    }
    expected = {
        "s2_only": S2_THRESHOLD,
        "gsv_mean": GSV_THRESHOLD,
        "fusion_mean": FUSION_THRESHOLD,
    }
    if frozen_thresholds != expected:
        raise ValueError(
            f"Frozen thresholds changed: {frozen_thresholds} != {expected}"
        )

    # Re-hash all model/preprocessing artifacts recorded at freeze time.
    frozen_hashes = protocol.get("frozen_artifact_hashes", {})
    required_names = {
        "s2_model",
        "s2_encoder",
        "gsv_mean_model",
        "gsv_mean_encoder",
        "fusion_mean_model",
        "fusion_train_only_scaler",
    }
    if not required_names.issubset(frozen_hashes):
        raise ValueError(
            "Freeze manifest is missing required artifact hashes: "
            f"{sorted(required_names - set(frozen_hashes))}"
        )

    for name in sorted(required_names):
        rec = frozen_hashes[name]
        path = Path(rec["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != rec["sha256"]:
            raise ValueError(
                f"Frozen artifact hash mismatch for {name}: "
                f"{actual} != {rec['sha256']}"
            )

    # Confirm paths match the final-test script.
    path_checks = {
        "s2_model": S2_MODEL_PATH,
        "s2_encoder": S2_ENCODER_PATH,
        "gsv_mean_model": GSV_MODEL_PATH,
        "gsv_mean_encoder": GSV_ENCODER_PATH,
        "fusion_mean_model": FUSION_MODEL_PATH,
        "fusion_train_only_scaler": FUSION_SCALER_PATH,
    }
    for name, expected_path in path_checks.items():
        if Path(frozen_hashes[name]["path"]) != expected_path:
            raise ValueError(f"Frozen path mismatch for {name}")

    routing = protocol["missing_gsv_policy"]["conditional_routing"]
    if routing.get("gsv_available") != "Fusion Mean":
        raise ValueError("Frozen covered-cell routing changed")
    if routing.get("gsv_unavailable") != "S2-only":
        raise ValueError("Frozen no-GSV routing changed")

    return protocol


def pr_auc(y, p):
    y = np.asarray(y, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    metric = tf.keras.metrics.AUC(curve="PR")
    metric.update_state(y, p)
    return float(metric.result().numpy())


def confusion_metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float32)
    pred = p >= threshold

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    nprec = tn / (tn + fn) if tn + fn else 0.0
    nrec = tn / (tn + fp) if tn + fp else 0.0
    nf1 = (
        2 * nprec * nrec / (nprec + nrec)
        if nprec + nrec
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "informal_precision": precision,
        "informal_recall": recall,
        "informal_f1": f1,
        "informal_iou": iou,
        "macro_f1": 0.5 * (f1 + nf1),
        "accuracy": (tp + tn) / len(y),
    }


def metric_row(scope, model, y, p, threshold, include_pr_auc=True):
    row = {
        "scope": scope,
        "model": model,
        "n": int(len(y)),
        "negative": int(np.sum(np.asarray(y) == 0)),
        "positive": int(np.sum(np.asarray(y) == 1)),
        "pr_auc": pr_auc(y, p) if include_pr_auc else "",
    }
    row.update(confusion_metrics(y, p, threshold))
    return row


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Test GSV construction
# Uses the same embedding-index grouping and deterministic slot ordering as
# the original GSV training code.
# =============================================================================

def read_test_image_rows():
    if not GSV_EMBED_INDEX.is_file():
        raise FileNotFoundError(GSV_EMBED_INDEX)

    with GSV_EMBED_INDEX.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])

        split_col = choose(fields, ["final_split", "split"], "split")
        label_col = choose(
            fields,
            ["final_label_numeric", "reference_label", "label"],
            "label",
        )
        row_col = choose(fields, ["global_cell_row"], "global_cell_row")
        col_col = choose(fields, ["global_cell_col"], "global_cell_col")
        fid_col = choose(fields, ["fid_numeric", "fid"], "fid")
        filename_col = choose(fields, ["filename"], "filename")
        path_col = choose(
            fields,
            [
                "resolved_relative_path",
                "image_path_relative_to_source",
                "resolved_image_path",
            ],
            "path",
        )

        rows = []
        seen_embedding_rows = set()
        total = 0

        for raw in reader:
            total += 1
            emb_row = parse_int(raw["embedding_row"], "embedding_row")
            if emb_row in seen_embedding_rows:
                raise ValueError(f"Duplicate embedding_row={emb_row}")
            seen_embedding_rows.add(emb_row)

            split = raw[split_col].strip().lower()
            if split != "test":
                continue

            label = parse_int(raw[label_col], label_col)
            if label not in {0, 1}:
                raise ValueError(f"Invalid test label={label}")

            rows.append(
                {
                    "split": split,
                    "label": label,
                    "global_cell_row": parse_int(raw[row_col], row_col),
                    "global_cell_col": parse_int(raw[col_col], col_col),
                    "fid_numeric": parse_int(raw[fid_col], fid_col),
                    "filename": raw[filename_col].strip(),
                    "relative_path": raw[path_col].strip(),
                    "embedding_row": emb_row,
                    "patch_order": parse_int(
                        raw.get("patch_order", ""),
                        "patch_order",
                    ),
                    "original_patch_index": parse_int(
                        raw.get("original_patch_index", ""),
                        "original_patch_index",
                    ),
                    "row_index_in_split": parse_int(
                        raw.get("row_index_in_split", ""),
                        "row_index_in_split",
                    ),
                    "cell_row": parse_int(
                        raw.get("cell_row", ""),
                        "cell_row",
                    ),
                    "cell_col": parse_int(
                        raw.get("cell_col", ""),
                        "cell_col",
                    ),
                    "informal_fraction": raw.get("informal_fraction", ""),
                }
            )

    if total != EXPECTED_IMAGES:
        raise ValueError(f"Embedding-index rows {total} != {EXPECTED_IMAGES}")
    if seen_embedding_rows != set(range(EXPECTED_IMAGES)):
        raise ValueError("embedding_row is not exact 0..34206")
    if len(rows) != EXPECTED_TEST_IMAGES:
        raise ValueError(
            f"Test GSV image rows {len(rows)} != {EXPECTED_TEST_IMAGES}"
        )

    return rows


def build_test_cell_records(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["global_cell_row"], r["global_cell_col"])].append(r)

    records = []

    for key in sorted(groups):
        group = groups[key]
        labels = {x["label"] for x in group}
        if len(labels) != 1:
            raise ValueError(
                f"Test cell {key} has inconsistent labels {labels}"
            )

        # Same deterministic rule used during original GSV training:
        # fid_numeric -> filename -> relative_path.
        group = sorted(
            group,
            key=lambda x: (
                x["fid_numeric"],
                x["filename"],
                x["relative_path"],
            ),
        )

        if not 1 <= len(group) <= K:
            raise ValueError(
                f"Test cell {key} has {len(group)} images; expected 1..{K}"
            )

        first = group[0]

        for item in group[1:]:
            for field in [
                "patch_order",
                "original_patch_index",
                "row_index_in_split",
                "cell_row",
                "cell_col",
            ]:
                if item[field] != first[field]:
                    raise ValueError(
                        f"Test cell {key}: inconsistent {field}"
                    )

        slot_rows = (
            [x["embedding_row"] for x in group]
            + [-1] * (K - len(group))
        )

        records.append(
            {
                "label": next(iter(labels)),
                "global_cell_row": key[0],
                "global_cell_col": key[1],
                "slot_count": len(group),
                "slot_rows": slot_rows,
                "patch_order": first["patch_order"],
                "original_patch_index": first["original_patch_index"],
                "row_index_in_split": first["row_index_in_split"],
                "cell_row": first["cell_row"],
                "cell_col": first["cell_col"],
                "informal_fraction": first["informal_fraction"],
            }
        )

    # Preserve deterministic test order compatible with S2 indexing.
    records.sort(
        key=lambda r: (
            r["patch_order"],
            r["cell_row"],
            r["cell_col"],
        )
    )

    if len(records) != EXPECTED_COVERED:
        raise ValueError(
            f"Covered test cells {len(records)} != {EXPECTED_COVERED}"
        )
    if sum(r["slot_count"] for r in records) != EXPECTED_TEST_IMAGES:
        raise ValueError("Covered-test slot image total mismatch")

    identities = [
        (r["patch_order"], r["cell_row"], r["cell_col"])
        for r in records
    ]
    if len(set(identities)) != EXPECTED_COVERED:
        raise ValueError("Duplicate covered test cell identity")

    return records


def records_to_gsv_inputs(records, embeddings):
    n = len(records)
    x = np.zeros((n, K, BASE_DIM), dtype=np.float32)
    mask = np.zeros((n, K), dtype=np.float32)

    for i, rec in enumerate(records):
        for slot, emb_row in enumerate(rec["slot_rows"]):
            if emb_row < 0:
                continue
            if not 0 <= emb_row < embeddings.shape[0]:
                raise ValueError(f"Invalid embedding row {emb_row}")
            x[i, slot] = embeddings[emb_row]
            mask[i, slot] = 1.0

    if not np.isfinite(x).all():
        raise ValueError("GSV test tensor contains non-finite values")

    counts = mask.sum(axis=1)
    if not np.all((counts >= 1) & (counts <= K)):
        raise ValueError("Invalid GSV slot mask in test")

    return x, mask


def gather_grid_values(grid, records):
    values = np.zeros((len(records),), dtype=np.float32)
    for i, rec in enumerate(records):
        values[i] = grid[
            rec["patch_order"],
            rec["cell_row"],
            rec["cell_col"],
        ]
    return values


def gather_grid_features(grid, records):
    out = np.zeros((len(records), FEATURE_DIM), dtype=np.float32)
    for i, rec in enumerate(records):
        out[i] = grid[
            rec["patch_order"],
            rec["cell_row"],
            rec["cell_col"],
        ]
    if not np.isfinite(out).all():
        raise ValueError("Gathered S2 test features contain non-finite values")
    return out


# =============================================================================
# Main — this is the first and only final-test evaluation
# =============================================================================

def main():
    ensure_clean_start()

    print("=" * 92)
    print("Nairobi v2 — ONE-TIME FINAL TEST — Seed 42")
    print("=" * 92)
    print("Primary seed:", SEED)
    print("No training: YES")
    print("No threshold selection: YES")
    print("No model selection: YES")
    print("Frozen thresholds:", S2_THRESHOLD, GSV_THRESHOLD, FUSION_THRESHOLD)
    print()

    protocol = verify_frozen_protocol()
    print("[1] Frozen protocol + artifact hashes: PASSED")
    print("    Model/threshold/routing changes since freeze: NONE")

    # ---------------------------------------------------------------------
    # FIRST TEST ACCESS begins here.
    # ---------------------------------------------------------------------
    print()
    print("[2] OPENING FINAL TEST DATA — first final-test evaluation")

    x_test = np.load(S2_TEST_INPUT, allow_pickle=False)
    y_patch = np.load(TEST_LABELS, allow_pickle=False)
    valid_mask = np.load(TEST_VALID_MASK, allow_pickle=False)

    if x_test.shape != (EXPECTED_TEST_PATCHES, 128, 128, 4):
        raise ValueError(f"Unexpected X_test shape {x_test.shape}")
    if x_test.dtype != np.float32 or not np.isfinite(x_test).all():
        raise ValueError("Invalid X_test")
    if y_patch.shape != (EXPECTED_TEST_PATCHES, 16, 16):
        raise ValueError(f"Unexpected test label shape {y_patch.shape}")
    if valid_mask.shape != (EXPECTED_TEST_PATCHES, 16, 16):
        raise ValueError(f"Unexpected test mask shape {valid_mask.shape}")

    valid_mask = valid_mask.astype(bool)
    y_patch = y_patch.astype(np.int8)

    if int(valid_mask.sum()) != EXPECTED_ALL_VALID:
        raise ValueError(
            f"Valid test cells {int(valid_mask.sum())} != {EXPECTED_ALL_VALID}"
        )

    y_all = y_patch[valid_mask]
    if (
        int(np.sum(y_all == 0)),
        int(np.sum(y_all == 1)),
    ) != (EXPECTED_ALL_NEG, EXPECTED_ALL_POS):
        raise ValueError("All-valid test class counts mismatch")

    print("    Test S2 patches:", x_test.shape[0])
    print("    All valid test cells:", int(valid_mask.sum()))
    print("    All-valid class counts:", EXPECTED_ALL_NEG, EXPECTED_ALL_POS)

    # ---------------------------------------------------------------------
    # S2-only test prediction
    # ---------------------------------------------------------------------
    s2_model = tf.keras.models.load_model(
        S2_MODEL_PATH,
        compile=False,
    )
    s2_encoder = tf.keras.models.load_model(
        S2_ENCODER_PATH,
        compile=False,
    )

    s2_probs = s2_model.predict(
        x_test,
        batch_size=16,
        verbose=0,
    ).astype(np.float32)
    if s2_probs.shape != (EXPECTED_TEST_PATCHES, 16, 16, 1):
        raise ValueError(f"Unexpected S2 prediction shape {s2_probs.shape}")
    if not np.isfinite(s2_probs).all():
        raise ValueError("Non-finite S2 test predictions")

    s2_probs_grid = s2_probs[..., 0]
    s2_features_grid = s2_encoder.predict(
        x_test,
        batch_size=16,
        verbose=0,
    ).astype(np.float32)
    if s2_features_grid.shape != (
        EXPECTED_TEST_PATCHES,
        16,
        16,
        FEATURE_DIM,
    ):
        raise ValueError(
            f"Unexpected S2 encoder output {s2_features_grid.shape}"
        )
    del x_test

    # ---------------------------------------------------------------------
    # Construct exact GSV-covered test cells using same rule as training.
    # ---------------------------------------------------------------------
    test_image_rows = read_test_image_rows()
    records = build_test_cell_records(test_image_rows)

    embeddings = np.load(
        GSV_EMBED_PATH,
        allow_pickle=False,
        mmap_mode="r",
    )
    if embeddings.shape != (EXPECTED_IMAGES, BASE_DIM):
        raise ValueError(f"Unexpected embedding shape {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise ValueError(f"Unexpected embedding dtype {embeddings.dtype}")

    gsv_x, gsv_mask = records_to_gsv_inputs(records, embeddings)
    del embeddings

    # Cross-check each covered-cell label against authoritative test labels.
    covered_y = np.zeros((EXPECTED_COVERED,), dtype=np.int8)
    covered_valid = np.zeros((EXPECTED_COVERED,), dtype=bool)

    for i, rec in enumerate(records):
        p, r, c = rec["patch_order"], rec["cell_row"], rec["cell_col"]
        if not (
            0 <= p < EXPECTED_TEST_PATCHES
            and 0 <= r < 16
            and 0 <= c < 16
        ):
            raise ValueError(f"Invalid covered test address {(p, r, c)}")
        covered_valid[i] = bool(valid_mask[p, r, c])
        covered_y[i] = int(y_patch[p, r, c])
        if covered_y[i] != rec["label"]:
            raise ValueError(
                f"GSV/S2 test label mismatch at {(p, r, c)}: "
                f"{rec['label']} vs {covered_y[i]}"
            )

    if not covered_valid.all():
        raise ValueError("A GSV-covered test record is not a valid test cell")

    no_gsv_count = int(valid_mask.sum()) - len(records)
    if no_gsv_count != EXPECTED_NO_GSV:
        raise ValueError(
            f"No-GSV test cells {no_gsv_count} != {EXPECTED_NO_GSV}"
        )

    covered_neg = int(np.sum(covered_y == 0))
    covered_pos = int(np.sum(covered_y == 1))

    print("[3] Covered-test mapping/alignment: PASSED")
    print("    GSV test images:", len(test_image_rows))
    print("    GSV-covered valid test cells:", len(records))
    print("    No-GSV valid test cells:", no_gsv_count)
    print("    Covered class counts:", covered_neg, covered_pos)

    # ---------------------------------------------------------------------
    # GSV Mean predictions and GSV cell features
    # ---------------------------------------------------------------------
    gsv_model = tf.keras.models.load_model(
        GSV_MODEL_PATH,
        custom_objects={"MaskedMeanPooling": MaskedMeanPooling},
        compile=False,
    )
    gsv_encoder = tf.keras.models.load_model(
        GSV_ENCODER_PATH,
        custom_objects={"MaskedMeanPooling": MaskedMeanPooling},
        compile=False,
    )

    gsv_probs = gsv_model.predict(
        {
            "gsv_embeddings_k7": gsv_x,
            "gsv_slot_mask": gsv_mask,
        },
        batch_size=128,
        verbose=0,
    ).astype(np.float32)

    if gsv_probs.shape != (EXPECTED_COVERED, 1):
        raise ValueError(f"Unexpected GSV probability shape {gsv_probs.shape}")
    if not np.isfinite(gsv_probs).all():
        raise ValueError("Non-finite GSV test probabilities")

    gsv_probs_1d = gsv_probs[:, 0]
    gsv_features = gsv_encoder.predict(
        {
            "gsv_embeddings_k7": gsv_x,
            "gsv_slot_mask": gsv_mask,
        },
        batch_size=128,
        verbose=0,
    ).astype(np.float32)

    if gsv_features.shape != (EXPECTED_COVERED, FEATURE_DIM):
        raise ValueError(f"Unexpected GSV feature shape {gsv_features.shape}")
    if not np.isfinite(gsv_features).all():
        raise ValueError("Non-finite GSV test features")

    del gsv_x, gsv_mask

    # ---------------------------------------------------------------------
    # Fusion predictions using frozen TRAIN-only scaler.
    # ---------------------------------------------------------------------
    covered_s2_probs = gather_grid_values(s2_probs_grid, records)
    covered_s2_features = gather_grid_features(
        s2_features_grid,
        records,
    )
    del s2_features_grid

    scaler = np.load(FUSION_SCALER_PATH, allow_pickle=False)
    required_scaler_keys = {"s2_mean", "s2_std", "gsv_mean", "gsv_std"}
    if set(scaler.files) != required_scaler_keys:
        raise ValueError(
            f"Unexpected Fusion scaler keys {set(scaler.files)}"
        )

    s2_mean = scaler["s2_mean"].astype(np.float32)
    s2_std = scaler["s2_std"].astype(np.float32)
    gsv_mean = scaler["gsv_mean"].astype(np.float32)
    gsv_std = scaler["gsv_std"].astype(np.float32)
    scaler.close()

    for name, arr in [
        ("s2_mean", s2_mean),
        ("s2_std", s2_std),
        ("gsv_mean", gsv_mean),
        ("gsv_std", gsv_std),
    ]:
        if arr.shape != (FEATURE_DIM,) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid scaler array {name}")

    covered_s2_z = (
        (covered_s2_features - s2_mean) / s2_std
    ).astype(np.float32)
    covered_gsv_z = (
        (gsv_features - gsv_mean) / gsv_std
    ).astype(np.float32)

    if not np.isfinite(covered_s2_z).all():
        raise ValueError("Non-finite standardized S2 test features")
    if not np.isfinite(covered_gsv_z).all():
        raise ValueError("Non-finite standardized GSV test features")

    fusion_model = tf.keras.models.load_model(
        FUSION_MODEL_PATH,
        compile=False,
    )
    fusion_probs = fusion_model.predict(
        {
            "s2_feature_256": covered_s2_z,
            "gsv_feature_256": covered_gsv_z,
        },
        batch_size=128,
        verbose=0,
    ).astype(np.float32)

    if fusion_probs.shape != (EXPECTED_COVERED, 1):
        raise ValueError(
            f"Unexpected Fusion probability shape {fusion_probs.shape}"
        )
    if not np.isfinite(fusion_probs).all():
        raise ValueError("Non-finite Fusion test probabilities")

    fusion_probs_1d = fusion_probs[:, 0]
    # ---------------------------------------------------------------------
    # Frozen-metric evaluation — NO threshold scanning.
    # ---------------------------------------------------------------------
    rows = []

    rows.append(
        metric_row(
            "GSV-covered test",
            "S2-only",
            covered_y,
            covered_s2_probs,
            S2_THRESHOLD,
            include_pr_auc=True,
        )
    )
    rows.append(
        metric_row(
            "GSV-covered test",
            "GSV Mean",
            covered_y,
            gsv_probs_1d,
            GSV_THRESHOLD,
            include_pr_auc=True,
        )
    )
    rows.append(
        metric_row(
            "GSV-covered test",
            "Fusion Mean",
            covered_y,
            fusion_probs_1d,
            FUSION_THRESHOLD,
            include_pr_auc=True,
        )
    )

    s2_all_probs = s2_probs_grid[valid_mask]
    rows.append(
        metric_row(
            "All-valid test",
            "S2-only everywhere",
            y_all,
            s2_all_probs,
            S2_THRESHOLD,
            include_pr_auc=True,
        )
    )

    # Conditional system: use model-specific frozen validation thresholds.
    # We intentionally do NOT compute a mixed-model PR-AUC.
    conditional_pred_grid = s2_probs_grid >= S2_THRESHOLD
    for i, rec in enumerate(records):
        conditional_pred_grid[
            rec["patch_order"],
            rec["cell_row"],
            rec["cell_col"],
        ] = fusion_probs_1d[i] >= FUSION_THRESHOLD

    conditional_pred_all = conditional_pred_grid[valid_mask].astype(np.int8)

    # Reuse confusion-metric helper using hard 0/1 predictions at threshold 0.5.
    conditional_metrics = confusion_metrics(
        y_all,
        conditional_pred_all.astype(np.float32),
        0.5,
    )
    conditional_row = {
        "scope": "All-valid test",
        "model": "Conditional: Fusion if GSV, otherwise S2",
        "n": int(len(y_all)),
        "negative": int(np.sum(y_all == 0)),
        "positive": int(np.sum(y_all == 1)),
        "pr_auc": "",
    }
    conditional_row.update(conditional_metrics)
    conditional_row["threshold"] = "routed: S2=0.78; Fusion=0.59"
    rows.append(conditional_row)

    # ---------------------------------------------------------------------
    # Save covered-cell predictions + identity for downstream XAI.
    # ---------------------------------------------------------------------
    covered_rows = []
    cell_index_rows = []

    for i, rec in enumerate(records):
        s2_pred = int(covered_s2_probs[i] >= S2_THRESHOLD)
        gsv_pred = int(gsv_probs_1d[i] >= GSV_THRESHOLD)
        fusion_pred = int(fusion_probs_1d[i] >= FUSION_THRESHOLD)

        base = {
            "covered_order": i,
            "patch_order": rec["patch_order"],
            "original_patch_index": rec["original_patch_index"],
            "row_index_in_split": rec["row_index_in_split"],
            "cell_row": rec["cell_row"],
            "cell_col": rec["cell_col"],
            "global_cell_row": rec["global_cell_row"],
            "global_cell_col": rec["global_cell_col"],
            "label": int(covered_y[i]),
            "slot_count": rec["slot_count"],
        }

        covered_rows.append(
            {
                **base,
                "s2_probability": float(covered_s2_probs[i]),
                "s2_prediction": s2_pred,
                "gsv_mean_probability": float(gsv_probs_1d[i]),
                "gsv_mean_prediction": gsv_pred,
                "fusion_mean_probability": float(fusion_probs_1d[i]),
                "fusion_mean_prediction": fusion_pred,
            }
        )

        slot_row = dict(base)
        for s, emb_row in enumerate(rec["slot_rows"], start=1):
            slot_row[f"slot_{s}_embedding_row"] = emb_row
        cell_index_rows.append(slot_row)

    # All-valid S2 + conditional hard prediction table.
    all_valid_rows = []
    for p in range(EXPECTED_TEST_PATCHES):
        for r in range(16):
            for c in range(16):
                if not valid_mask[p, r, c]:
                    continue
                all_valid_rows.append(
                    {
                        "patch_order": p,
                        "cell_row": r,
                        "cell_col": c,
                        "label": int(y_patch[p, r, c]),
                        "s2_probability": float(s2_probs_grid[p, r, c]),
                        "s2_prediction": int(
                            s2_probs_grid[p, r, c] >= S2_THRESHOLD
                        ),
                        "conditional_prediction": int(
                            conditional_pred_grid[p, r, c]
                        ),
                    }
                )

    if len(all_valid_rows) != EXPECTED_ALL_VALID:
        raise ValueError("All-valid prediction row count mismatch")
    # ---------------------------------------------------------------------
    # Final audit
    # ---------------------------------------------------------------------
    covered_prevalence = covered_pos / EXPECTED_COVERED
    no_gsv_y = np.ones(EXPECTED_ALL_VALID, dtype=bool)
    # Build no-GSV prevalence without relying on mixed probability routing.
    covered_identity_set = {
        (r["patch_order"], r["cell_row"], r["cell_col"])
        for r in records
    }
    no_gsv_labels = []
    for p in range(EXPECTED_TEST_PATCHES):
        for r in range(16):
            for c in range(16):
                if not valid_mask[p, r, c]:
                    continue
                if (p, r, c) not in covered_identity_set:
                    no_gsv_labels.append(int(y_patch[p, r, c]))
    no_gsv_labels = np.asarray(no_gsv_labels, dtype=np.int8)
    if len(no_gsv_labels) != EXPECTED_NO_GSV:
        raise ValueError("No-GSV label count mismatch")

    # ---------------------------------------------------------------------
    # Commit final-test outputs only after ALL prediction/alignment checks pass.
    # This avoids leaving partial protected outputs if a technical check fails.
    # ---------------------------------------------------------------------
    np.save(S2_TEST_PROBS_GRID, s2_probs_grid, allow_pickle=False)
    np.save(
        GSV_TEST_PROBS_COVERED,
        gsv_probs_1d,
        allow_pickle=False,
    )
    np.save(
        FUSION_TEST_PROBS_COVERED,
        fusion_probs_1d,
        allow_pickle=False,
    )
    write_rows(METRICS_CSV, rows)
    write_rows(COVERED_PRED_CSV, covered_rows)
    write_rows(COVERED_CELL_INDEX, cell_index_rows)
    write_rows(ALL_VALID_PRED_CSV, all_valid_rows)

    audit = {
        "status": "PASSED",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "One-time final test",
        "primary_seed": SEED,
        "protocol_manifest": str(PROTOCOL_MANIFEST),
        "protocol_status_verified": "FROZEN",
        "frozen_artifact_hashes_reverified": True,
        "training_performed": False,
        "threshold_selection_performed": False,
        "model_selection_performed": False,
        "thresholds": {
            "s2_only": S2_THRESHOLD,
            "gsv_mean": GSV_THRESHOLD,
            "fusion_mean": FUSION_THRESHOLD,
        },
        "test_population": {
            "test_patches": EXPECTED_TEST_PATCHES,
            "all_valid_cells": EXPECTED_ALL_VALID,
            "all_valid_negative": EXPECTED_ALL_NEG,
            "all_valid_positive": EXPECTED_ALL_POS,
            "gsv_covered_valid_cells": EXPECTED_COVERED,
            "gsv_covered_negative": covered_neg,
            "gsv_covered_positive": covered_pos,
            "gsv_covered_prevalence": covered_prevalence,
            "no_gsv_valid_cells": EXPECTED_NO_GSV,
            "no_gsv_negative": int(np.sum(no_gsv_labels == 0)),
            "no_gsv_positive": int(np.sum(no_gsv_labels == 1)),
            "no_gsv_prevalence": float(np.mean(no_gsv_labels == 1)),
            "gsv_test_images": EXPECTED_TEST_IMAGES,
        },
        "metrics": rows,
        "conditional_mixed_model_pr_auc_reported": False,
        "post_test_retraining_or_tuning_allowed": False,
        "outputs": {
            "metrics_csv": str(METRICS_CSV),
            "covered_predictions_csv": str(COVERED_PRED_CSV),
            "all_valid_predictions_csv": str(ALL_VALID_PRED_CSV),
            "covered_cell_index_csv": str(COVERED_CELL_INDEX),
            "s2_probability_grid_npy": str(S2_TEST_PROBS_GRID),
            "gsv_probability_covered_npy": str(GSV_TEST_PROBS_COVERED),
            "fusion_probability_covered_npy": str(FUSION_TEST_PROBS_COVERED),
        },
    }
    AUDIT_JSON.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    def fmt(x):
        return f"{x:.6f}" if isinstance(x, float) else str(x)

    lines = [
        "=" * 92,
        "Nairobi v2 — ONE-TIME FINAL TEST AUDIT — Seed 42",
        "=" * 92,
        "STATUS: PASSED",
        "Protocol status before test: FROZEN",
        "Frozen artifact hashes reverified: PASSED",
        "Training performed: NO",
        "Threshold selection on test: NO",
        "Model selection on test: NO",
        "",
        "Test population:",
        f"  All valid cells: {EXPECTED_ALL_VALID} "
        f"(negative={EXPECTED_ALL_NEG}, positive={EXPECTED_ALL_POS})",
        f"  GSV-covered valid cells: {EXPECTED_COVERED} "
        f"(negative={covered_neg}, positive={covered_pos})",
        f"  No-GSV valid cells: {EXPECTED_NO_GSV} "
        f"(negative={int(np.sum(no_gsv_labels == 0))}, "
        f"positive={int(np.sum(no_gsv_labels == 1))})",
        f"  GSV images in covered test cells: {EXPECTED_TEST_IMAGES}",
        "",
        "Frozen thresholds:",
        f"  S2-only: {S2_THRESHOLD}",
        f"  GSV Mean: {GSV_THRESHOLD}",
        f"  Fusion Mean: {FUSION_THRESHOLD}",
        "",
        "Final test metrics:",
    ]

    for row in rows:
        lines.extend(
            [
                f"  [{row['scope']}] {row['model']}",
                f"    PR-AUC: {fmt(row['pr_auc'])}",
                f"    Informal precision: {fmt(row['informal_precision'])}",
                f"    Informal recall: {fmt(row['informal_recall'])}",
                f"    Informal F1: {fmt(row['informal_f1'])}",
                f"    Informal IoU: {fmt(row['informal_iou'])}",
                f"    Macro-F1: {fmt(row['macro_f1'])}",
                f"    Accuracy: {fmt(row['accuracy'])}",
            ]
        )

    lines.extend(
        [
            "",
            "Conditional mixed-model PR-AUC: NOT REPORTED",
            "Post-test retraining/tuning: PROHIBITED",
            "",
            "ONE-TIME FINAL TEST: COMPLETED",
            f"Metrics: {METRICS_CSV}",
            f"Audit: {AUDIT_JSON}",
        ]
    )

    AUDIT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
