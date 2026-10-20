#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
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
EMBED_DIR = PROJECT / "scratch/features/gsv_embeddings/resnet50_imagenet_v2"
EMBED_PATH = EMBED_DIR / "gsv_resnet50_embeddings_v2.npy"
INDEX_PATH = EMBED_DIR / "gsv_resnet50_embedding_index_v2.csv"

SEED = int(os.environ.get("GSV_SEED", "42"))
BATCH_SIZE = int(os.environ.get("GSV_CELL_BATCH_SIZE", "128"))
MAX_EPOCHS = int(os.environ.get("GSV_MAX_EPOCHS", "100"))
LR = float(os.environ.get("GSV_LR", "1e-4"))
EARLY_STOP_PATIENCE = int(os.environ.get("GSV_EARLY_STOP_PATIENCE", "12"))
LR_PATIENCE = int(os.environ.get("GSV_LR_PATIENCE", "5"))
DROPOUT = float(os.environ.get("GSV_DROPOUT", "0.30"))

K = 7
BASE_DIM = 2048
REP_DIM = 256
ATT_DIM = 128

EXPECTED_IMAGES = 34207
EXPECTED_TRAIN_IMAGES = 20282
EXPECTED_VAL_IMAGES = 6250
EXPECTED_TRAIN_CELLS = 15481
EXPECTED_VAL_CELLS = 4902

MODEL_DIR = PROJECT / f"models/gsv_only_v2/seed_{SEED}"
RESULT_DIR = PROJECT / f"results/gsv_only_v2/seed_{SEED}"
CKPT_DIR = PROJECT / f"scratch/checkpoints/gsv_only_v2/seed_{SEED}"

BEST_WEIGHTS = CKPT_DIR / "best.weights.h5"
FULL_MODEL = MODEL_DIR / "gsv_only_attention_best_v2.keras"
ENCODER_MODEL = MODEL_DIR / "gsv_encoder_attention_best_v2.keras"
ATTENTION_MODEL = MODEL_DIR / "gsv_attention_weights_best_v2.keras"

HISTORY_CSV = RESULT_DIR / "training_history_v2.csv"
THRESHOLD_CSV = RESULT_DIR / "val_threshold_scan_v2.csv"
VAL_PROBS = RESULT_DIR / "val_probabilities_v2.npy"
VAL_ATTENTION = RESULT_DIR / "val_attention_weights_v2.npy"
CELL_INDEX = RESULT_DIR / "gsv_train_val_cell_index_v2.csv"
CONFIG_JSON = RESULT_DIR / "gsv_only_training_config_v2.json"
AUDIT_JSON = RESULT_DIR / "gsv_only_training_audit_v2.json"
AUDIT_TXT = RESULT_DIR / "gsv_only_training_audit_v2.txt"


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_int(value, name):
    value = str(value).strip()
    if not value:
        raise ValueError(f"Empty {name}")
    return int(float(value))


def choose(fieldnames, candidates, what):
    for name in candidates:
        if name in fieldnames:
            return name
    raise ValueError(f"Missing {what}; columns={fieldnames}")


def ensure_clean_start():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p for p in [FULL_MODEL, ENCODER_MODEL, AUDIT_JSON] if p.exists()]
    if existing:
        raise FileExistsError(
            "Protected completed outputs already exist:\n"
            + "\n".join(map(str, existing))
        )


def load_embeddings():
    if not EMBED_PATH.is_file():
        raise FileNotFoundError(EMBED_PATH)
    x = np.load(EMBED_PATH, allow_pickle=False)
    if x.shape != (EXPECTED_IMAGES, BASE_DIM):
        raise ValueError(f"Embedding shape {x.shape} is invalid")
    if x.dtype != np.float32:
        raise ValueError(f"Embedding dtype {x.dtype} is invalid")
    if not np.isfinite(x).all():
        raise ValueError("Embeddings contain non-finite values")
    print("Embedding table:", x.shape, x.dtype)
    return x


def read_train_val_rows():
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(INDEX_PATH)

    with INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as f:
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
        split_counts = defaultdict(int)

        for raw in reader:
            total += 1
            emb_row = parse_int(raw["embedding_row"], "embedding_row")
            if emb_row in seen_embedding_rows:
                raise ValueError(f"Duplicate embedding_row={emb_row}")
            seen_embedding_rows.add(emb_row)

            split = raw[split_col].strip().lower()

            # Test rows are deliberately ignored before label/cell sample creation.
            if split not in {"train", "val"}:
                continue

            label = parse_int(raw[label_col], label_col)
            if label not in {0, 1}:
                raise ValueError(f"Invalid {split} label={label}")

            item = {
                "split": split,
                "label": label,
                "global_cell_row": parse_int(raw[row_col], row_col),
                "global_cell_col": parse_int(raw[col_col], col_col),
                "fid_numeric": parse_int(raw[fid_col], fid_col),
                "filename": raw[filename_col].strip(),
                "relative_path": raw[path_col].strip(),
                "embedding_row": emb_row,
                "patch_order": raw.get("patch_order", ""),
                "original_patch_index": raw.get("original_patch_index", ""),
                "row_index_in_split": raw.get("row_index_in_split", ""),
                "cell_row": raw.get("cell_row", ""),
                "cell_col": raw.get("cell_col", ""),
                "informal_fraction": raw.get("informal_fraction", ""),
            }
            rows.append(item)
            split_counts[split] += 1

    if total != EXPECTED_IMAGES:
        raise ValueError(f"Index rows {total} != {EXPECTED_IMAGES}")
    if seen_embedding_rows != set(range(EXPECTED_IMAGES)):
        raise ValueError("embedding_row is not an exact 0..34206 permutation")
    if split_counts["train"] != EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            f"Train images {split_counts['train']} != {EXPECTED_TRAIN_IMAGES}"
        )
    if split_counts["val"] != EXPECTED_VAL_IMAGES:
        raise ValueError(
            f"Val images {split_counts['val']} != {EXPECTED_VAL_IMAGES}"
        )

    print("Embedding-index validation: PASSED")
    print("Train/val image rows:", split_counts["train"], split_counts["val"])
    return rows


def build_cell_records(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[
            (r["split"], r["global_cell_row"], r["global_cell_col"])
        ].append(r)

    records = []
    for key in sorted(groups):
        group = groups[key]
        labels = {x["label"] for x in group}
        if len(labels) != 1:
            raise ValueError(f"Cell {key} has inconsistent labels {labels}")

        # Same deterministic slot rule used in the aligned GSV metadata.
        group = sorted(
            group,
            key=lambda x: (
                x["fid_numeric"],
                x["filename"],
                x["relative_path"],
            ),
        )
        if not 1 <= len(group) <= K:
            raise ValueError(f"Cell {key} has {len(group)} images")

        first = group[0]
        slot_rows = [x["embedding_row"] for x in group] + [-1] * (K - len(group))

        records.append(
            {
                "split": key[0],
                "label": next(iter(labels)),
                "global_cell_row": key[1],
                "global_cell_col": key[2],
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

    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]

    if len(train) != EXPECTED_TRAIN_CELLS:
        raise ValueError(f"Train covered cells {len(train)} != {EXPECTED_TRAIN_CELLS}")
    if len(val) != EXPECTED_VAL_CELLS:
        raise ValueError(f"Val covered cells {len(val)} != {EXPECTED_VAL_CELLS}")
    if sum(r["slot_count"] for r in train) != EXPECTED_TRAIN_IMAGES:
        raise ValueError("Train slot-image total mismatch")
    if sum(r["slot_count"] for r in val) != EXPECTED_VAL_IMAGES:
        raise ValueError("Val slot-image total mismatch")

    print("Covered-cell grouping: PASSED")
    print("Train covered cells/images:", len(train), EXPECTED_TRAIN_IMAGES)
    print("Val covered cells/images:", len(val), EXPECTED_VAL_IMAGES)
    return train, val


def records_to_arrays(records, embeddings):
    n = len(records)
    x = np.zeros((n, K, BASE_DIM), dtype=np.float32)
    m = np.zeros((n, K), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)

    for i, rec in enumerate(records):
        y[i, 0] = rec["label"]
        for slot, emb_row in enumerate(rec["slot_rows"]):
            if emb_row >= 0:
                x[i, slot] = embeddings[emb_row]
                m[i, slot] = 1.0

    if not np.isfinite(x).all():
        raise ValueError("Cell tensor contains non-finite values")
    if not np.all((m.sum(axis=1) >= 1) & (m.sum(axis=1) <= K)):
        raise ValueError("Invalid GSV slot mask")
    return x, m, y


def write_cell_index(train, val):
    fields = [
        "split",
        "cell_order_within_split",
        "label",
        "global_cell_row",
        "global_cell_col",
        "patch_order",
        "original_patch_index",
        "row_index_in_split",
        "cell_row",
        "cell_col",
        "informal_fraction",
        "slot_count",
    ] + [f"slot_{i+1}_embedding_row" for i in range(K)]

    with CELL_INDEX.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, records in [("train", train), ("val", val)]:
            for order, rec in enumerate(records):
                row = {
                    "split": split,
                    "cell_order_within_split": order,
                    "label": rec["label"],
                    "global_cell_row": rec["global_cell_row"],
                    "global_cell_col": rec["global_cell_col"],
                    "patch_order": rec["patch_order"],
                    "original_patch_index": rec["original_patch_index"],
                    "row_index_in_split": rec["row_index_in_split"],
                    "cell_row": rec["cell_row"],
                    "cell_col": rec["cell_col"],
                    "informal_fraction": rec["informal_fraction"],
                    "slot_count": rec["slot_count"],
                }
                for i, emb_row in enumerate(rec["slot_rows"]):
                    row[f"slot_{i+1}_embedding_row"] = emb_row
                writer.writerow(row)


@tf.keras.utils.register_keras_serializable(package="NairobiV2")
class MaskedGatedAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dim=ATT_DIM, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = int(attention_dim)
        self.v = tf.keras.layers.Dense(self.attention_dim, activation="tanh")
        self.u = tf.keras.layers.Dense(self.attention_dim, activation="sigmoid")
        self.w = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, inputs):
        z, slot_mask = inputs
        scores = tf.squeeze(self.w(self.v(z) * self.u(z)), axis=-1)
        mask = tf.cast(slot_mask, scores.dtype)
        scores = tf.where(mask > 0, scores, tf.cast(-1e9, scores.dtype))
        alpha = tf.nn.softmax(scores, axis=1)
        alpha = alpha * mask
        alpha = tf.math.divide_no_nan(
            alpha,
            tf.reduce_sum(alpha, axis=1, keepdims=True),
        )
        pooled = tf.reduce_sum(z * alpha[..., None], axis=1)
        return pooled, alpha

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"attention_dim": self.attention_dim})
        return cfg


@tf.keras.utils.register_keras_serializable(package="NairobiV2")
class PositiveWeightedBCE(tf.keras.losses.Loss):
    def __init__(self, positive_weight, name="positive_weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.positive_weight = float(positive_weight)

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        return (
            -self.positive_weight * y_true * tf.math.log(y_pred)
            -(1.0-y_true) * tf.math.log(1.0-y_pred)
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"positive_weight": self.positive_weight})
        return cfg


def build_models():
    emb_in = tf.keras.Input((K, BASE_DIM), name="gsv_embeddings_k7")
    mask_in = tf.keras.Input((K,), name="gsv_slot_mask")

    z = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            REP_DIM,
            activation="relu",
            kernel_initializer="he_normal",
        ),
        name="shared_projection_2048_to_256",
    )(emb_in)

    z = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dropout(DROPOUT),
        name="shared_projection_dropout",
    )(z)

    pooled, alpha = MaskedGatedAttention(
        ATT_DIM,
        name="masked_gated_attention",
    )([z, mask_in])

    rep = tf.keras.layers.LayerNormalization(
        name="gsv_representation_256"
    )(pooled)

    h = tf.keras.layers.Dropout(
        DROPOUT,
        name="gsv_head_dropout",
    )(rep)

    prob = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="gsv_probability",
    )(h)

    full = tf.keras.Model([emb_in, mask_in], prob, name="nairobi_v2_gsv_only")
    encoder = tf.keras.Model(
        [emb_in, mask_in],
        rep,
        name="nairobi_v2_gsv_encoder",
    )
    attention = tf.keras.Model(
        [emb_in, mask_in],
        alpha,
        name="nairobi_v2_gsv_attention",
    )
    return full, encoder, attention


def confusion(y, p, t):
    pred = p >= t
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    iou = tp/(tp+fp+fn) if tp+fp+fn else 0.0
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    nprec = tn/(tn+fn) if tn+fn else 0.0
    nrec = tn/(tn+fp) if tn+fp else 0.0
    nf1 = 2*nprec*nrec/(nprec+nrec) if nprec+nrec else 0.0

    return {
        "threshold": float(t),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "informal_precision": precision,
        "informal_recall": recall,
        "informal_f1": f1,
        "informal_iou": iou,
        "macro_f1": 0.5*(f1+nf1),
        "accuracy": accuracy,
    }


def threshold_scan(y, p):
    rows = [
        confusion(y, p, float(t))
        for t in np.round(np.arange(0.05, 0.9501, 0.005), 3)
    ]
    best = max(
        rows,
        key=lambda r: (
            r["informal_f1"],
            r["macro_f1"],
            -abs(r["threshold"]-0.5),
            -r["threshold"],
        ),
    )
    return rows, best


def pr_auc(y, p):
    m = tf.keras.metrics.AUC(curve="PR")
    m.update_state(y.astype(np.float32), p.astype(np.float32))
    return float(m.result().numpy())


def main():
    seed_everything(SEED)
    ensure_clean_start()

    print("="*80)
    print("Nairobi v2 — GSV-only masked-attention training")
    print("="*80)
    print("Python:", sys.executable)
    print("TensorFlow:", tf.__version__)
    print("Seed:", SEED)
    print("Batch size:", BATCH_SIZE)
    print("Max epochs:", MAX_EPOCHS)
    print("Learning rate:", LR)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    gpus = tf.config.list_physical_devices("GPU")
    print("Physical GPUs:", gpus)
    if not gpus:
        raise RuntimeError("TensorFlow detected no GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print("Logical GPUs:", tf.config.list_logical_devices("GPU"))

    embeddings = load_embeddings()
    rows = read_train_val_rows()
    train_records, val_records = build_cell_records(rows)
    write_cell_index(train_records, val_records)

    x_train, m_train, y_train = records_to_arrays(train_records, embeddings)
    x_val, m_val, y_val = records_to_arrays(val_records, embeddings)
    del embeddings

    train_neg = int(np.sum(y_train[:, 0] == 0))
    train_pos = int(np.sum(y_train[:, 0] == 1))
    val_neg = int(np.sum(y_val[:, 0] == 0))
    val_pos = int(np.sum(y_val[:, 0] == 1))
    if train_pos == 0 or val_pos == 0:
        raise ValueError("Positive class missing")

    positive_weight = train_neg / train_pos

    print("Cell tensor construction: PASSED")
    print("X_train:", x_train.shape, x_train.dtype)
    print("X_val:", x_val.shape, x_val.dtype)
    print("Train class counts:", train_neg, train_pos)
    print("Val class counts:", val_neg, val_pos)
    print(
        "Train slot count min/max/mean:",
        int(m_train.sum(1).min()),
        int(m_train.sum(1).max()),
        float(m_train.sum(1).mean()),
    )
    print(
        "Val slot count min/max/mean:",
        int(m_val.sum(1).min()),
        int(m_val.sum(1).max()),
        float(m_val.sum(1).mean()),
    )
    print("Positive-class weight:", positive_weight)
    print("GSV-only population: GSV-COVERED VALID CELLS ONLY")
    print("TEST DATA EVALUATED: NO")

    model, encoder, attention_model = build_models()
    if model.output_shape != (None, 1):
        raise ValueError(model.output_shape)
    if encoder.output_shape != (None, REP_DIM):
        raise ValueError(encoder.output_shape)
    if attention_model.output_shape != (None, K):
        raise ValueError(attention_model.output_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=PositiveWeightedBCE(positive_weight),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
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
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(HISTORY_CSV), append=False),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    start = time.time()
    history = model.fit(
        {"gsv_embeddings_k7": x_train, "gsv_slot_mask": m_train},
        y_train,
        validation_data=(
            {"gsv_embeddings_k7": x_val, "gsv_slot_mask": m_val},
            y_val,
        ),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start

    model.load_weights(BEST_WEIGHTS)

    val_probs = model.predict(
        {"gsv_embeddings_k7": x_val, "gsv_slot_mask": m_val},
        batch_size=BATCH_SIZE,
        verbose=0,
    ).astype(np.float32)

    val_attention = attention_model.predict(
        {"gsv_embeddings_k7": x_val, "gsv_slot_mask": m_val},
        batch_size=BATCH_SIZE,
        verbose=0,
    ).astype(np.float32)

    if not np.isfinite(val_probs).all():
        raise ValueError("Non-finite validation probabilities")
    if not np.isfinite(val_attention).all():
        raise ValueError("Non-finite validation attention")
    if not np.allclose(val_attention.sum(1), 1.0, atol=1e-5):
        raise ValueError("Attention does not sum to one")
    if (m_val == 0).any() and not np.allclose(
        val_attention[m_val == 0],
        0.0,
        atol=1e-7,
    ):
        raise ValueError("Padded slots have non-zero attention")

    np.save(VAL_PROBS, val_probs, allow_pickle=False)
    np.save(VAL_ATTENTION, val_attention, allow_pickle=False)

    yv = y_val[:, 0].astype(np.int8)
    pv = val_probs[:, 0]

    scan, best = threshold_scan(yv, pv)
    with THRESHOLD_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scan[0].keys()))
        writer.writeheader()
        writer.writerows(scan)

    at05 = confusion(yv, pv, 0.5)
    val_pr = pr_auc(yv, pv)
    brier = float(np.mean((pv - yv.astype(np.float32))**2))

    model.save(FULL_MODEL)
    encoder.save(ENCODER_MODEL)
    attention_model.save(ATTENTION_MODEL)

    epochs_ran = len(history.history["loss"])
    best_epoch = int(np.argmax(history.history["val_pr_auc"])) + 1
    completed = datetime.now().isoformat()

    config = {
        "created_at": completed,
        "seed": SEED,
        "tensorflow_version": tf.__version__,
        "base_embeddings": str(EMBED_PATH),
        "embedding_index": str(INDEX_PATH),
        "population": "valid 80m cells with >=1 GSV image only",
        "train_covered_cells": len(train_records),
        "val_covered_cells": len(val_records),
        "train_images": EXPECTED_TRAIN_IMAGES,
        "val_images": EXPECTED_VAL_IMAGES,
        "train_class_counts": {"negative": train_neg, "positive": train_pos},
        "val_class_counts": {"negative": val_neg, "positive": val_pos},
        "positive_weight": positive_weight,
        "architecture": {
            "input": [K, BASE_DIM],
            "shared_projection": "Dense 2048->256 ReLU + Dropout",
            "aggregation": "masked gated attention over real slots only",
            "representation": REP_DIM,
            "classifier": "LayerNorm -> Dropout -> Dense(1,sigmoid)",
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "optimizer": "Adam",
            "initial_lr": LR,
            "monitor": "val_pr_auc",
            "elapsed_seconds": elapsed,
        },
        "threshold_selection": "validation covered cells only",
        "test_data_evaluated": False,
    }
    atomic_json(CONFIG_JSON, config)

    audit = {
        "created_at": completed,
        "status": "PASSED",
        "epochs_ran": epochs_ran,
        "best_epoch_by_val_pr_auc": best_epoch,
        "val_pr_auc": val_pr,
        "val_brier": brier,
        "val_metrics_threshold_0_5": at05,
        "selected_validation_threshold": best,
        "attention_sum_check": True,
        "padded_attention_zero_check": True,
        "test_data_evaluated": False,
    }
    atomic_json(AUDIT_JSON, audit)

    lines = [
        "="*80,
        "Nairobi v2 — GSV-only Attention Training Audit",
        "="*80,
        f"Seed: {SEED}",
        f"Train covered cells/images: {len(train_records)}/{EXPECTED_TRAIN_IMAGES}",
        f"Validation covered cells/images: {len(val_records)}/{EXPECTED_VAL_IMAGES}",
        f"Train class counts: negative={train_neg}, positive={train_pos}",
        f"Validation class counts: negative={val_neg}, positive={val_pos}",
        f"Positive-class weight: {positive_weight}",
        "Base ResNet50 embeddings: FROZEN",
        "Shared trainable projection: 2048 -> 256",
        "Aggregation: MASKED GATED ATTENTION, K<=7",
        "Padded-slot attention: EXACT ZERO",
        "Attention weights sum to 1 over real slots: PASSED",
        "GSV-only population: GSV-COVERED VALID CELLS ONLY",
        f"Epochs run: {epochs_ran}",
        f"Best epoch by validation PR-AUC: {best_epoch}",
        f"Validation PR-AUC: {val_pr}",
        f"Validation Brier score: {brier}",
        f"Validation @0.5 informal F1: {at05['informal_f1']}",
        f"Selected validation threshold: {best['threshold']}",
        f"Selected-threshold informal precision: {best['informal_precision']}",
        f"Selected-threshold informal recall: {best['informal_recall']}",
        f"Selected-threshold informal F1: {best['informal_f1']}",
        f"Selected-threshold informal IoU: {best['informal_iou']}",
        f"Selected-threshold macro-F1: {best['macro_f1']}",
        "TEST DATA EVALUATED: NO",
        "",
        "GSV-ONLY ATTENTION TRAINING: PASSED",
    ]
    AUDIT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print("Saved full model:", FULL_MODEL)
    print("Saved GSV encoder:", ENCODER_MODEL)
    print("Saved attention model:", ATTENTION_MODEL)
    print("Saved history:", HISTORY_CSV)
    print("Saved threshold scan:", THRESHOLD_CSV)
    print("Saved validation probabilities:", VAL_PROBS)
    print("Saved validation attention:", VAL_ATTENTION)
    print("Saved cell index:", CELL_INDEX)
    print("Saved config:", CONFIG_JSON)
    print("Saved audit JSON:", AUDIT_JSON)
    print("Saved audit TXT:", AUDIT_TXT)


if __name__ == "__main__":
    main()
