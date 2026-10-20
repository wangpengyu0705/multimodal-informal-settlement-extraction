#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf


PROJECT = Path.home() / "projects" / "nairobi_v2"

SEED = int(os.environ.get("FUSION_MEAN_SEED", "42"))
BATCH_SIZE = 128
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
DROPOUT = 0.30
HIDDEN_UNITS = 128
L2_WEIGHT = 1e-4
EARLY_PATIENCE = 12
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-6

K = 7
BASE_DIM = 2048
FEATURE_DIM = 256

EXPECTED_IMAGES = 34207
EXPECTED_TRAIN_PATCHES = 306
EXPECTED_VAL_PATCHES = 102
EXPECTED_TRAIN = 15481
EXPECTED_VAL = 4902
EXPECTED_TRAIN_NEG = 12920
EXPECTED_TRAIN_POS = 2561
EXPECTED_VAL_NEG = 4037
EXPECTED_VAL_POS = 865

S2_TRAIN_PATH = PROJECT / "scratch/features/s2_inputs/X_train_s2_v2.npy"
S2_VAL_PATH = PROJECT / "scratch/features/s2_inputs/X_val_s2_v2.npy"

S2_ENCODER_PATH = (
    PROJECT / f"models/s2_only_v2/seed_{SEED}/s2_encoder_best_v2.keras"
)
S2_VAL_PROBS_PATH = (
    PROJECT / f"results/s2_only_v2/seed_{SEED}/val_probabilities_v2.npy"
)

GSV_ENCODER_PATH = (
    PROJECT / f"models/gsv_mean_pool_v2/seed_{SEED}/gsv_encoder_mean_pool_best_v2.keras"
)
GSV_VAL_PROBS_PATH = (
    PROJECT / f"results/gsv_mean_pool_v2/seed_{SEED}/val_probabilities_v2.npy"
)
GSV_EMBED_PATH = (
    PROJECT
    / "scratch/features/gsv_embeddings/resnet50_imagenet_v2/"
      "gsv_resnet50_embeddings_v2.npy"
)

# The covered-cell mapping is metadata-like and seed-independent.
CELL_INDEX_PATH = (
    PROJECT / "results/gsv_only_v2/seed_42/gsv_train_val_cell_index_v2.csv"
)

FEATURE_DIR = (
    PROJECT / f"scratch/features/fusion_inputs_mean_pool_v2/seed_{SEED}"
)
MODEL_DIR = PROJECT / f"models/fusion_mean_pool_v2/seed_{SEED}"
CKPT_DIR = (
    PROJECT / f"scratch/checkpoints/fusion_mean_pool_v2/seed_{SEED}"
)
RESULT_DIR = PROJECT / f"results/fusion_mean_pool_v2/seed_{SEED}"

TRAIN_S2_FEATURES = FEATURE_DIR / "train_s2_features_covered_v2.npy"
VAL_S2_FEATURES = FEATURE_DIR / "val_s2_features_covered_v2.npy"
TRAIN_GSV_FEATURES = FEATURE_DIR / "train_gsv_mean_features_covered_v2.npy"
VAL_GSV_FEATURES = FEATURE_DIR / "val_gsv_mean_features_covered_v2.npy"
TRAIN_LABELS = FEATURE_DIR / "train_labels_covered_v2.npy"
VAL_LABELS = FEATURE_DIR / "val_labels_covered_v2.npy"

BEST_WEIGHTS = CKPT_DIR / "best.weights.h5"
MODEL_PATH = MODEL_DIR / "fusion_mean_pool_best_v2.keras"
SCALER_PATH = RESULT_DIR / "fusion_mean_pool_standardization_train_only_v2.npz"
VAL_PROBS_PATH = RESULT_DIR / "val_probabilities_v2.npy"
HISTORY_PATH = RESULT_DIR / "training_history_v2.csv"
THRESHOLD_PATH = RESULT_DIR / "val_threshold_scan_v2.csv"
AUDIT_JSON = RESULT_DIR / "fusion_mean_pool_training_audit_v2.json"
AUDIT_TXT = RESULT_DIR / "fusion_mean_pool_training_audit_v2.txt"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@tf.keras.utils.register_keras_serializable(package="NairobiV2")
class MaskedMeanPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        z, slot_mask = inputs
        mask = tf.cast(slot_mask, z.dtype)[..., None]
        numerator = tf.reduce_sum(z * mask, axis=1)
        denominator = tf.reduce_sum(mask, axis=1)
        return tf.math.divide_no_nan(numerator, denominator)


@tf.keras.utils.register_keras_serializable(package="NairobiV2")
class PositiveWeightedBCE(tf.keras.losses.Loss):
    def __init__(self, pos_weight, name="positive_weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = float(pos_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        tf.debugging.assert_equal(
            tf.shape(y_true),
            tf.shape(y_pred),
            message="PositiveWeightedBCE requires identical y_true/y_pred shapes",
        )
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss = -(
            self.pos_weight * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"pos_weight": self.pos_weight})
        return cfg


def ensure_clean_start():
    for d in (FEATURE_DIR, MODEL_DIR, CKPT_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    protected = [
        TRAIN_S2_FEATURES,
        VAL_S2_FEATURES,
        TRAIN_GSV_FEATURES,
        VAL_GSV_FEATURES,
        MODEL_PATH,
        AUDIT_JSON,
    ]
    existing = [p for p in protected if p.exists()]
    if existing:
        raise FileExistsError(
            "Protected outputs already exist; refusing overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )


def to_int(value, name):
    value = str(value).strip()
    if value == "":
        raise ValueError(f"Empty integer field: {name}")
    return int(float(value))


def read_cell_records():
    if not CELL_INDEX_PATH.is_file():
        raise FileNotFoundError(CELL_INDEX_PATH)

    records = {"train": [], "val": []}
    with CELL_INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row["split"]).strip().lower()
            if split not in records:
                continue
            rec = {
                "order": to_int(
                    row["cell_order_within_split"],
                    "cell_order_within_split",
                ),
                "label": to_int(row["label"], "label"),
                "patch_order": to_int(row["patch_order"], "patch_order"),
                "cell_row": to_int(row["cell_row"], "cell_row"),
                "cell_col": to_int(row["cell_col"], "cell_col"),
                "slot_count": to_int(row["slot_count"], "slot_count"),
                "slot_rows": [
                    to_int(
                        row[f"slot_{i+1}_embedding_row"],
                        f"slot_{i+1}_embedding_row",
                    )
                    for i in range(K)
                ],
            }
            records[split].append(rec)

    for split, expected in (("train", EXPECTED_TRAIN), ("val", EXPECTED_VAL)):
        records[split].sort(key=lambda r: r["order"])
        if len(records[split]) != expected:
            raise ValueError(f"{split} cell count {len(records[split])} != {expected}")
        if [r["order"] for r in records[split]] != list(range(expected)):
            raise ValueError(f"{split} cell ordering is not exact 0..{expected-1}")

        for rec in records[split]:
            active = sum(int(v >= 0) for v in rec["slot_rows"])
            if active != rec["slot_count"] or not 1 <= active <= K:
                raise ValueError("GSV slot-count integrity failure")

    return records


def load_s2_input(path: Path, n: int):
    if not path.is_file():
        raise FileNotFoundError(path)
    x = np.load(path, allow_pickle=False)
    if x.shape != (n, 128, 128, 4) or x.dtype != np.float32:
        raise ValueError(f"Unexpected S2 input: {path} {x.shape} {x.dtype}")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite S2 inputs")
    return x


def gather_s2(features, records, expected_patches):
    out = np.zeros((len(records), FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((len(records),), dtype=np.int8)
    for i, rec in enumerate(records):
        p, r, c = rec["patch_order"], rec["cell_row"], rec["cell_col"]
        if not (0 <= p < expected_patches and 0 <= r < 16 and 0 <= c < 16):
            raise ValueError(f"Invalid S2 cell address: {(p, r, c)}")
        out[i] = features[p, r, c]
        labels[i] = rec["label"]
    if not np.isfinite(out).all():
        raise ValueError("Non-finite gathered S2 features")
    return out, labels


def gather_s2_probabilities(prob_grid, records, expected_patches):
    arr = np.asarray(prob_grid)
    if arr.shape == (expected_patches, 16, 16, 1):
        arr = arr[..., 0]
    if arr.shape != (expected_patches, 16, 16):
        raise ValueError(f"Unexpected S2 probability shape: {arr.shape}")

    out = np.zeros((len(records),), dtype=np.float32)
    for i, rec in enumerate(records):
        p, r, c = rec["patch_order"], rec["cell_row"], rec["cell_col"]
        out[i] = arr[p, r, c]
    if not np.isfinite(out).all():
        raise ValueError("Non-finite matched S2 probabilities")
    return out


def extract_gsv_features(encoder, records, embedding_table, batch=256):
    out = np.zeros((len(records), FEATURE_DIM), dtype=np.float32)

    for start in range(0, len(records), batch):
        stop = min(start + batch, len(records))
        part = records[start:stop]
        n = len(part)

        x = np.zeros((n, K, BASE_DIM), dtype=np.float32)
        mask = np.zeros((n, K), dtype=np.float32)

        for j, rec in enumerate(part):
            for slot, emb_row in enumerate(rec["slot_rows"]):
                if emb_row < 0:
                    continue
                if not 0 <= emb_row < embedding_table.shape[0]:
                    raise ValueError(f"Invalid embedding row {emb_row}")
                x[j, slot] = embedding_table[emb_row]
                mask[j, slot] = 1.0

        pred = encoder.predict(
            {"gsv_embeddings_k7": x, "gsv_slot_mask": mask},
            batch_size=batch,
            verbose=0,
        ).astype(np.float32)

        if pred.shape != (n, FEATURE_DIM):
            raise ValueError(f"Unexpected GSV encoder output {pred.shape}")
        out[start:stop] = pred

    if not np.isfinite(out).all():
        raise ValueError("Non-finite GSV features")
    return out


def fit_standardizer(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    near_zero = int(np.sum(std < 1e-6))
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std, near_zero


def standardize(x, mean, std):
    z = ((x - mean) / std).astype(np.float32)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite standardized features")
    return z


def build_model():
    s2_in = tf.keras.Input((FEATURE_DIM,), name="s2_feature_256")
    gsv_in = tf.keras.Input((FEATURE_DIM,), name="gsv_feature_256")
    x = tf.keras.layers.Concatenate(name="feature_concatenation")([s2_in, gsv_in])
    x = tf.keras.layers.Dense(
        HIDDEN_UNITS,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
        name="fusion_dense_128",
    )(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="fusion_dropout")(x)
    out = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="informal_probability",
    )(x)
    return tf.keras.Model(
        [s2_in, gsv_in],
        out,
        name="nairobi_v2_final_mean_pool_fusion",
    )


def pr_auc(y, p):
    m = tf.keras.metrics.AUC(curve="PR")
    m.update_state(
        np.asarray(y, dtype=np.float32),
        np.asarray(p, dtype=np.float32),
    )
    return float(m.result().numpy())


def confusion(y, p, threshold):
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


def scan_thresholds(y, p):
    rows = [
        confusion(y, p, float(t))
        for t in np.round(np.arange(0.05, 0.9501, 0.005), 3)
    ]
    best = max(
        rows,
        key=lambda r: (
            r["informal_f1"],
            r["macro_f1"],
            -abs(r["threshold"] - 0.5),
            -r["threshold"],
        ),
    )
    return rows, best


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed(SEED)
    ensure_clean_start()

    print("=" * 88)
    print("Nairobi v2 — final mean-pooling Fusion multiseed training")
    print("=" * 88)
    print("Seed:", SEED)
    print("Python:", sys.executable)
    print("TensorFlow:", tf.__version__)
    print("Physical GPUs:", tf.config.list_physical_devices("GPU"))
    print("TEST DATA LOADED/EVALUATED: NO")

    required = [
        S2_ENCODER_PATH,
        S2_VAL_PROBS_PATH,
        GSV_ENCODER_PATH,
        GSV_VAL_PROBS_PATH,
        GSV_EMBED_PATH,
        CELL_INDEX_PATH,
        S2_TRAIN_PATH,
        S2_VAL_PATH,
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(str(p) for p in missing)
        )

    records = read_cell_records()
    y_train = np.array([r["label"] for r in records["train"]], dtype=np.int8)
    y_val = np.array([r["label"] for r in records["val"]], dtype=np.int8)

    train_counts = (int(np.sum(y_train == 0)), int(np.sum(y_train == 1)))
    val_counts = (int(np.sum(y_val == 0)), int(np.sum(y_val == 1)))
    if train_counts != (EXPECTED_TRAIN_NEG, EXPECTED_TRAIN_POS):
        raise ValueError(f"Train class counts {train_counts}")
    if val_counts != (EXPECTED_VAL_NEG, EXPECTED_VAL_POS):
        raise ValueError(f"Val class counts {val_counts}")

    # Seed-specific frozen S2 encoder.
    s2_encoder = tf.keras.models.load_model(S2_ENCODER_PATH, compile=False)
    if s2_encoder.output_shape != (None, 16, 16, FEATURE_DIM):
        raise ValueError(f"Unexpected S2 encoder output {s2_encoder.output_shape}")

    x_train_s2 = load_s2_input(S2_TRAIN_PATH, EXPECTED_TRAIN_PATCHES)
    x_val_s2 = load_s2_input(S2_VAL_PATH, EXPECTED_VAL_PATCHES)

    s2_train_grid = s2_encoder.predict(
        x_train_s2, batch_size=16, verbose=0
    ).astype(np.float32)
    s2_val_grid = s2_encoder.predict(
        x_val_s2, batch_size=16, verbose=0
    ).astype(np.float32)
    del x_train_s2, x_val_s2

    if s2_train_grid.shape != (EXPECTED_TRAIN_PATCHES, 16, 16, FEATURE_DIM):
        raise ValueError(s2_train_grid.shape)
    if s2_val_grid.shape != (EXPECTED_VAL_PATCHES, 16, 16, FEATURE_DIM):
        raise ValueError(s2_val_grid.shape)

    train_s2, y_train_s2 = gather_s2(
        s2_train_grid, records["train"], EXPECTED_TRAIN_PATCHES
    )
    val_s2, y_val_s2 = gather_s2(
        s2_val_grid, records["val"], EXPECTED_VAL_PATCHES
    )
    del s2_train_grid, s2_val_grid

    if not np.array_equal(y_train, y_train_s2):
        raise ValueError("Train S2/cell-index label mismatch")
    if not np.array_equal(y_val, y_val_s2):
        raise ValueError("Val S2/cell-index label mismatch")

    # Seed-specific frozen GSV mean-pooling encoder.
    gsv_encoder = tf.keras.models.load_model(
        GSV_ENCODER_PATH,
        custom_objects={"MaskedMeanPooling": MaskedMeanPooling},
        compile=False,
    )
    if gsv_encoder.output_shape != (None, FEATURE_DIM):
        raise ValueError(f"Unexpected GSV encoder output {gsv_encoder.output_shape}")

    embeddings = np.load(
        GSV_EMBED_PATH,
        allow_pickle=False,
        mmap_mode="r",
    )
    if embeddings.shape != (EXPECTED_IMAGES, BASE_DIM):
        raise ValueError(f"Unexpected embedding table {embeddings.shape}")

    train_gsv = extract_gsv_features(
        gsv_encoder, records["train"], embeddings
    )
    val_gsv = extract_gsv_features(
        gsv_encoder, records["val"], embeddings
    )
    del embeddings

    for name, arr, shape in [
        ("train_s2", train_s2, (EXPECTED_TRAIN, FEATURE_DIM)),
        ("val_s2", val_s2, (EXPECTED_VAL, FEATURE_DIM)),
        ("train_gsv", train_gsv, (EXPECTED_TRAIN, FEATURE_DIM)),
        ("val_gsv", val_gsv, (EXPECTED_VAL, FEATURE_DIM)),
    ]:
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"{name}: invalid {arr.shape}")

    np.save(TRAIN_S2_FEATURES, train_s2, allow_pickle=False)
    np.save(VAL_S2_FEATURES, val_s2, allow_pickle=False)
    np.save(TRAIN_GSV_FEATURES, train_gsv, allow_pickle=False)
    np.save(VAL_GSV_FEATURES, val_gsv, allow_pickle=False)
    np.save(TRAIN_LABELS, y_train, allow_pickle=False)
    np.save(VAL_LABELS, y_val, allow_pickle=False)

    print("Seed-specific S2/GSV frozen feature extraction: PASSED")
    print("Exact shared covered-cell order/labels: PASSED")

    # Same-seed matched validation baselines; no retraining.
    s2_val_grid_probs = np.load(S2_VAL_PROBS_PATH, allow_pickle=False)
    matched_s2_probs = gather_s2_probabilities(
        s2_val_grid_probs,
        records["val"],
        EXPECTED_VAL_PATCHES,
    )
    matched_s2_pr = pr_auc(y_val, matched_s2_probs)

    gsv_probs = np.load(GSV_VAL_PROBS_PATH, allow_pickle=False)
    if gsv_probs.shape == (EXPECTED_VAL, 1):
        gsv_probs = gsv_probs[:, 0]
    if gsv_probs.shape != (EXPECTED_VAL,) or not np.isfinite(gsv_probs).all():
        raise ValueError(f"Unexpected GSV val probabilities {gsv_probs.shape}")
    matched_gsv_pr = pr_auc(y_val, gsv_probs)

    s2_mean, s2_std, s2_zero = fit_standardizer(train_s2)
    gsv_mean, gsv_std, gsv_zero = fit_standardizer(train_gsv)
    train_s2_z = standardize(train_s2, s2_mean, s2_std)
    val_s2_z = standardize(val_s2, s2_mean, s2_std)
    train_gsv_z = standardize(train_gsv, gsv_mean, gsv_std)
    val_gsv_z = standardize(val_gsv, gsv_mean, gsv_std)

    np.savez(
        SCALER_PATH,
        s2_mean=s2_mean,
        s2_std=s2_std,
        gsv_mean=gsv_mean,
        gsv_std=gsv_std,
    )

    pos_weight = EXPECTED_TRAIN_NEG / EXPECTED_TRAIN_POS
    y_train_fit = y_train.astype(np.float32)[:, None]
    y_val_fit = y_val.astype(np.float32)[:, None]

    probe = PositiveWeightedBCE(pos_weight).call(
        tf.constant([[0.0], [1.0]], dtype=tf.float32),
        tf.constant([[0.25], [0.75]], dtype=tf.float32),
    )
    if tuple(probe.shape) != (2,):
        raise ValueError(f"Unexpected loss probe shape {probe.shape}")

    print("Train/val cells:", EXPECTED_TRAIN, EXPECTED_VAL)
    print("Train class counts:", train_counts)
    print("Val class counts:", val_counts)
    print("Positive weight:", pos_weight)
    print("S2 near-zero std dimensions:", s2_zero)
    print("GSV near-zero std dimensions:", gsv_zero)
    print("Feature standardization fitted on TRAIN ONLY: PASSED")
    print("Custom weighted-BCE shape contract: PASSED")
    print("Matched S2 PR-AUC:", matched_s2_pr)
    print("Matched GSV Mean PR-AUC:", matched_gsv_pr)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=PositiveWeightedBCE(pos_weight),
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_WEIGHTS),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=EARLY_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            patience=LR_PATIENCE,
            factor=LR_FACTOR,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        {"s2_feature_256": train_s2_z, "gsv_feature_256": train_gsv_z},
        y_train_fit,
        validation_data=(
            {"s2_feature_256": val_s2_z, "gsv_feature_256": val_gsv_z},
            y_val_fit,
        ),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - t0

    model.load_weights(BEST_WEIGHTS)

    history_rows = []
    for i in range(len(history.history["loss"])):
        row = {"epoch": i + 1}
        for key, vals in history.history.items():
            row[key] = float(vals[i])
        history_rows.append(row)
    write_csv(HISTORY_PATH, history_rows)

    val_probs = model.predict(
        {"s2_feature_256": val_s2_z, "gsv_feature_256": val_gsv_z},
        batch_size=BATCH_SIZE,
        verbose=0,
    ).astype(np.float32)
    if val_probs.shape != (EXPECTED_VAL, 1) or not np.isfinite(val_probs).all():
        raise ValueError(f"Invalid fusion validation probabilities {val_probs.shape}")
    np.save(VAL_PROBS_PATH, val_probs, allow_pickle=False)

    fusion_pr = pr_auc(y_val, val_probs[:, 0])
    scan_rows, best = scan_thresholds(y_val, val_probs[:, 0])
    write_csv(THRESHOLD_PATH, scan_rows)
    metrics_05 = confusion(y_val, val_probs[:, 0], 0.5)

    best_epoch = int(np.argmax(history.history["val_pr_auc"]) + 1)
    model.save(MODEL_PATH)

    audit = {
        "status": "PASSED",
        "experiment": "Final mean-pooling Fusion multiseed training",
        "seed": SEED,
        "population": "GSV-covered valid cells only",
        "architecture": (
            "seed-specific frozen S2-256 + seed-specific frozen mean-pooled "
            "GSV-256 -> separate train-only standardization -> concat-512 -> "
            "Dense128 ReLU L2 -> Dropout0.30 -> sigmoid"
        ),
        "train_cells": EXPECTED_TRAIN,
        "val_cells": EXPECTED_VAL,
        "train_class_counts": {
            "negative": EXPECTED_TRAIN_NEG,
            "positive": EXPECTED_TRAIN_POS,
        },
        "val_class_counts": {
            "negative": EXPECTED_VAL_NEG,
            "positive": EXPECTED_VAL_POS,
        },
        "trainable_fusion_parameters": int(model.count_params()),
        "best_epoch": best_epoch,
        "validation_pr_auc": fusion_pr,
        "matched_s2_validation_pr_auc": matched_s2_pr,
        "matched_gsv_mean_validation_pr_auc": matched_gsv_pr,
        "fusion_minus_matched_s2_pr_auc": fusion_pr - matched_s2_pr,
        "fusion_minus_gsv_mean_pr_auc": fusion_pr - matched_gsv_pr,
        "metrics_at_0_5": metrics_05,
        "selected_validation_threshold": best,
        "feature_standardization_fit": "TRAIN ONLY",
        "test_data_loaded": False,
        "test_data_evaluated": False,
        "elapsed_seconds_training": elapsed,
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "=" * 88,
        "Nairobi v2 — Final Mean-Pooling Fusion Multiseed Training Audit",
        "=" * 88,
        f"Seed: {SEED}",
        "Population: GSV-COVERED VALID CELLS ONLY",
        f"Train cells: {EXPECTED_TRAIN}",
        f"Validation cells: {EXPECTED_VAL}",
        f"Trainable Fusion parameters: {model.count_params()}",
        f"Best epoch: {best_epoch}",
        f"Validation PR-AUC: {fusion_pr}",
        f"Matched S2 validation PR-AUC: {matched_s2_pr}",
        f"GSV Mean validation PR-AUC: {matched_gsv_pr}",
        f"Fusion - matched S2 PR-AUC: {fusion_pr - matched_s2_pr}",
        f"Fusion - GSV Mean PR-AUC: {fusion_pr - matched_gsv_pr}",
        f"F1 @ 0.5: {metrics_05['informal_f1']}",
        f"Selected validation threshold: {best['threshold']}",
        f"Selected-threshold precision: {best['informal_precision']}",
        f"Selected-threshold recall: {best['informal_recall']}",
        f"Selected-threshold F1: {best['informal_f1']}",
        f"Selected-threshold IoU: {best['informal_iou']}",
        f"Selected-threshold macro-F1: {best['macro_f1']}",
        "Feature standardization fitted on TRAIN ONLY: PASSED",
        "Exact seed-specific S2/GSV covered-cell alignment: PASSED",
        "",
        "TEST DATA LOADED: NO",
        "TEST DATA EVALUATED: NO",
        "",
        "FINAL MEAN-POOLING FUSION MULTISEED TRAINING: PASSED",
    ]
    AUDIT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print("Saved S2 features:", TRAIN_S2_FEATURES, VAL_S2_FEATURES)
    print("Saved GSV features:", TRAIN_GSV_FEATURES, VAL_GSV_FEATURES)
    print("Saved Fusion model:", MODEL_PATH)
    print("Saved scaler:", SCALER_PATH)
    print("Saved validation probabilities:", VAL_PROBS_PATH)
    print("Saved audit:", AUDIT_TXT)


if __name__ == "__main__":
    main()
