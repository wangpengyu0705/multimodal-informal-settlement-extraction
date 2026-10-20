#!/usr/bin/env python3
from __future__ import annotations

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


# =============================================================================
# Fixed project paths
# =============================================================================

PROJECT = Path.home() / "projects" / "nairobi_v2"

S2_INPUT_DIR = PROJECT / "scratch/features/s2_inputs"
LABEL_DIR = PROJECT / "metadata/latest_aligned_v2"

X_TRAIN_PATH = S2_INPUT_DIR / "X_train_s2_v2.npy"
X_VAL_PATH = S2_INPUT_DIR / "X_val_s2_v2.npy"

Y_TRAIN_PATH = LABEL_DIR / "train_label_patches_v2.npy"
Y_VAL_PATH = LABEL_DIR / "val_label_patches_v2.npy"

MASK_TRAIN_PATH = LABEL_DIR / "train_valid_label_mask_v2.npy"
MASK_VAL_PATH = LABEL_DIR / "val_valid_label_mask_v2.npy"

# IMPORTANT:
# Test data are intentionally not loaded or referenced by this training script.

SEED = int(os.environ.get("S2_SEED", "42"))
BATCH_SIZE = int(os.environ.get("S2_BATCH_SIZE", "8"))
MAX_EPOCHS = int(os.environ.get("S2_MAX_EPOCHS", "100"))
LEARNING_RATE = float(os.environ.get("S2_LR", "1e-4"))
EARLY_STOP_PATIENCE = int(
    os.environ.get("S2_EARLY_STOP_PATIENCE", "12")
)
LR_PATIENCE = int(os.environ.get("S2_LR_PATIENCE", "5"))
DROPOUT_RATE = float(os.environ.get("S2_DROPOUT", "0.30"))

MODEL_DIR = PROJECT / f"models/s2_only_v2/seed_{SEED}"
RESULT_DIR = PROJECT / f"results/s2_only_v2/seed_{SEED}"
CHECKPOINT_DIR = (
    PROJECT / f"scratch/checkpoints/s2_only_v2/seed_{SEED}"
)

BEST_WEIGHTS = CHECKPOINT_DIR / "best.weights.h5"
FULL_MODEL_PATH = MODEL_DIR / "s2_only_model_best_v2.keras"
ENCODER_PATH = MODEL_DIR / "s2_encoder_best_v2.keras"

HISTORY_CSV = RESULT_DIR / "training_history_v2.csv"
THRESHOLD_CSV = RESULT_DIR / "val_threshold_scan_v2.csv"
VAL_PROBS_NPY = RESULT_DIR / "val_probabilities_v2.npy"
CONFIG_JSON = RESULT_DIR / "s2_only_training_config_v2.json"
AUDIT_JSON = RESULT_DIR / "s2_only_training_audit_v2.json"
AUDIT_TXT = RESULT_DIR / "s2_only_training_audit_v2.txt"

EXPECTED = {
    "train_shape": (306, 128, 128, 4),
    "val_shape": (102, 128, 128, 4),
    "train_label_shape": (306, 16, 16),
    "val_label_shape": (102, 16, 16),
    "train_valid": 67220,
    "val_valid": 23024,
    "train_negative": 62787,
    "train_positive": 4433,
    "val_negative": 21566,
    "val_positive": 1458,
}


# =============================================================================
# Reproducibility and utilities
# =============================================================================

def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def atomic_write_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def ensure_clean_start() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    protected_outputs = [
        FULL_MODEL_PATH,
        ENCODER_PATH,
        AUDIT_JSON,
    ]

    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "This seed already has protected completed outputs. "
            "Refusing to overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )


# =============================================================================
# Data validation
# =============================================================================

def load_and_validate_data():
    required = [
        X_TRAIN_PATH,
        X_VAL_PATH,
        Y_TRAIN_PATH,
        Y_VAL_PATH,
        MASK_TRAIN_PATH,
        MASK_VAL_PATH,
    ]

    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(str(path) for path in missing)
        )

    x_train = np.load(X_TRAIN_PATH, allow_pickle=False)
    x_val = np.load(X_VAL_PATH, allow_pickle=False)

    y_train_raw = np.load(Y_TRAIN_PATH, allow_pickle=False)
    y_val_raw = np.load(Y_VAL_PATH, allow_pickle=False)

    mask_train = np.load(
        MASK_TRAIN_PATH,
        allow_pickle=False,
    ).astype(bool, copy=False)

    mask_val = np.load(
        MASK_VAL_PATH,
        allow_pickle=False,
    ).astype(bool, copy=False)

    expected_pairs = [
        ("X_train", x_train.shape, EXPECTED["train_shape"]),
        ("X_val", x_val.shape, EXPECTED["val_shape"]),
        (
            "y_train",
            y_train_raw.shape,
            EXPECTED["train_label_shape"],
        ),
        (
            "y_val",
            y_val_raw.shape,
            EXPECTED["val_label_shape"],
        ),
        (
            "mask_train",
            mask_train.shape,
            EXPECTED["train_label_shape"],
        ),
        (
            "mask_val",
            mask_val.shape,
            EXPECTED["val_label_shape"],
        ),
    ]

    for name, observed, expected in expected_pairs:
        if observed != expected:
            raise ValueError(
                f"{name} shape {observed} != {expected}"
            )

    if x_train.dtype != np.float32:
        raise ValueError(
            f"X_train dtype must be float32, got {x_train.dtype}"
        )
    if x_val.dtype != np.float32:
        raise ValueError(
            f"X_val dtype must be float32, got {x_val.dtype}"
        )

    if not np.isfinite(x_train).all():
        raise ValueError("X_train contains non-finite values")
    if not np.isfinite(x_val).all():
        raise ValueError("X_val contains non-finite values")

    x_min = min(float(x_train.min()), float(x_val.min()))
    x_max = max(float(x_train.max()), float(x_val.max()))

    if x_min < -1e-6 or x_max > 1.0 + 1e-6:
        raise ValueError(
            f"S2 inputs outside [0,1]: min={x_min}, max={x_max}"
        )

    if not np.array_equal(mask_train, y_train_raw != -1):
        raise ValueError(
            "Train valid mask does not exactly equal (label != -1)"
        )
    if not np.array_equal(mask_val, y_val_raw != -1):
        raise ValueError(
            "Validation valid mask does not exactly equal (label != -1)"
        )

    train_valid = int(mask_train.sum())
    val_valid = int(mask_val.sum())

    if train_valid != EXPECTED["train_valid"]:
        raise ValueError(
            f"Train valid count {train_valid} "
            f"!= {EXPECTED['train_valid']}"
        )
    if val_valid != EXPECTED["val_valid"]:
        raise ValueError(
            f"Validation valid count {val_valid} "
            f"!= {EXPECTED['val_valid']}"
        )

    train_valid_labels = y_train_raw[mask_train]
    val_valid_labels = y_val_raw[mask_val]

    if not np.isin(train_valid_labels, [0, 1]).all():
        raise ValueError(
            "Train valid labels contain values outside {0,1}"
        )
    if not np.isin(val_valid_labels, [0, 1]).all():
        raise ValueError(
            "Validation valid labels contain values outside {0,1}"
        )

    counts = {
        "train_valid": train_valid,
        "val_valid": val_valid,
        "train_negative": int(
            np.sum(train_valid_labels == 0)
        ),
        "train_positive": int(
            np.sum(train_valid_labels == 1)
        ),
        "val_negative": int(
            np.sum(val_valid_labels == 0)
        ),
        "val_positive": int(
            np.sum(val_valid_labels == 1)
        ),
    }

    for key in [
        "train_negative",
        "train_positive",
        "val_negative",
        "val_positive",
    ]:
        if counts[key] != EXPECTED[key]:
            raise ValueError(
                f"{key} {counts[key]} != {EXPECTED[key]}"
            )

    # Invalid labels are replaced by 0 only in the packed tensor.
    # Their mask is 0 and they are excluded from BOTH numerator and
    # denominator of the custom loss and from all metrics.
    y_train_clean = np.where(
        mask_train,
        y_train_raw,
        0,
    ).astype(np.float32)

    y_val_clean = np.where(
        mask_val,
        y_val_raw,
        0,
    ).astype(np.float32)

    # Pack [label, valid_mask] in the last dimension so that the custom
    # training model has explicit access to the mask when computing loss.
    y_train_packed = np.stack(
        [
            y_train_clean,
            mask_train.astype(np.float32),
        ],
        axis=-1,
    )

    y_val_packed = np.stack(
        [
            y_val_clean,
            mask_val.astype(np.float32),
        ],
        axis=-1,
    )

    print("Data validation: PASSED")
    print("X_train:", x_train.shape, x_train.dtype)
    print("X_val:", x_val.shape, x_val.dtype)
    print(
        "Train valid/class counts:",
        counts["train_valid"],
        counts["train_negative"],
        counts["train_positive"],
    )
    print(
        "Val valid/class counts:",
        counts["val_valid"],
        counts["val_negative"],
        counts["val_positive"],
    )
    print(
        "S2 input range:",
        f"min={x_min:.6f}",
        f"max={x_max:.6f}",
    )

    return (
        x_train,
        y_train_packed,
        x_val,
        y_val_packed,
        y_val_raw,
        mask_val,
        counts,
    )


# =============================================================================
# Architecture
# =============================================================================

def conv_block(x, filters: int, block_name: str):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{block_name}_conv1",
    )(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{block_name}_conv2",
    )(x)

    return x


def build_core_model_and_encoder():
    inputs = tf.keras.Input(
        shape=(128, 128, 4),
        name="s2_input",
    )

    x = conv_block(inputs, 64, "enc_block1")
    x = tf.keras.layers.MaxPooling2D(
        pool_size=2,
        name="enc_pool1",
    )(x)

    x = conv_block(x, 128, "enc_block2")
    x = tf.keras.layers.MaxPooling2D(
        pool_size=2,
        name="enc_pool2",
    )(x)

    x = conv_block(x, 256, "enc_block3")
    encoder_output = tf.keras.layers.MaxPooling2D(
        pool_size=2,
        name="s2_encoder_16x16x256",
    )(x)

    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="s2_head_dropout",
    )(encoder_output)

    probabilities = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="s2_probability",
    )(x)

    core_model = tf.keras.Model(
        inputs=inputs,
        outputs=probabilities,
        name="nairobi_v2_s2_only",
    )

    encoder = tf.keras.Model(
        inputs=inputs,
        outputs=encoder_output,
        name="nairobi_v2_s2_encoder",
    )

    return core_model, encoder


# =============================================================================
# Strict valid-cell-normalized training wrapper
# =============================================================================

@tf.keras.utils.register_keras_serializable(
    package="NairobiV2"
)
class MaskedS2TrainingModel(tf.keras.Model):
    """
    Training wrapper around the functional S2 model.

    Loss is EXACTLY:

        sum_valid(weighted_BCE) / number_of_valid_cells

    Invalid cells therefore contribute neither to the numerator nor to
    the denominator. Metrics are also updated only on valid cells.
    """

    def __init__(
        self,
        core_model: tf.keras.Model,
        positive_weight: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.core_model = core_model
        self.positive_weight = float(positive_weight)

        self.loss_tracker = tf.keras.metrics.Mean(
            name="loss"
        )
        self.accuracy_metric = (
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
                threshold=0.5,
            )
        )
        self.precision_metric = tf.keras.metrics.Precision(
            name="precision",
            thresholds=0.5,
        )
        self.recall_metric = tf.keras.metrics.Recall(
            name="recall",
            thresholds=0.5,
        )
        self.pr_auc_metric = tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR",
        )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.accuracy_metric,
            self.precision_metric,
            self.recall_metric,
            self.pr_auc_metric,
        ]

    def call(self, inputs, training=False):
        return self.core_model(
            inputs,
            training=training,
        )

    def _unpack_target(self, y_packed):
        labels = tf.cast(
            y_packed[..., 0:1],
            tf.float32,
        )
        valid_mask = tf.cast(
            y_packed[..., 1:2],
            tf.float32,
        )
        return labels, valid_mask

    def _masked_weighted_bce(
        self,
        labels,
        predictions,
        valid_mask,
    ):
        eps = tf.keras.backend.epsilon()
        predictions = tf.clip_by_value(
            predictions,
            eps,
            1.0 - eps,
        )

        per_cell = (
            -self.positive_weight
            * labels
            * tf.math.log(predictions)
            - (1.0 - labels)
            * tf.math.log(1.0 - predictions)
        )

        numerator = tf.reduce_sum(
            per_cell * valid_mask
        )
        denominator = tf.reduce_sum(valid_mask)

        loss = tf.math.divide_no_nan(
            numerator,
            denominator,
        )

        return loss, denominator

    def _update_masked_metrics(
        self,
        labels,
        predictions,
        valid_mask,
    ):
        flat_labels = tf.reshape(labels, [-1])
        flat_predictions = tf.reshape(
            predictions,
            [-1],
        )
        flat_mask = tf.reshape(valid_mask, [-1])

        self.accuracy_metric.update_state(
            flat_labels,
            flat_predictions,
            sample_weight=flat_mask,
        )
        self.precision_metric.update_state(
            flat_labels,
            flat_predictions,
            sample_weight=flat_mask,
        )
        self.recall_metric.update_state(
            flat_labels,
            flat_predictions,
            sample_weight=flat_mask,
        )
        self.pr_auc_metric.update_state(
            flat_labels,
            flat_predictions,
            sample_weight=flat_mask,
        )

    def train_step(self, data):
        x, y_packed = data
        labels, valid_mask = self._unpack_target(
            y_packed
        )

        with tf.GradientTape() as tape:
            predictions = self(
                x,
                training=True,
            )

            loss, valid_count = (
                self._masked_weighted_bce(
                    labels,
                    predictions,
                    valid_mask,
                )
            )

            if self.core_model.losses:
                loss += tf.add_n(
                    self.core_model.losses
                )

        gradients = tape.gradient(
            loss,
            self.core_model.trainable_weights,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.core_model.trainable_weights,
            )
        )

        # Weight batch loss by number of valid cells so epoch loss is also
        # a true valid-cell-weighted mean across variable-validity batches.
        self.loss_tracker.update_state(
            loss,
            sample_weight=valid_count,
        )

        self._update_masked_metrics(
            labels,
            predictions,
            valid_mask,
        )

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }

    def test_step(self, data):
        x, y_packed = data
        labels, valid_mask = self._unpack_target(
            y_packed
        )

        predictions = self(
            x,
            training=False,
        )

        loss, valid_count = self._masked_weighted_bce(
            labels,
            predictions,
            valid_mask,
        )

        if self.core_model.losses:
            loss += tf.add_n(
                self.core_model.losses
            )

        self.loss_tracker.update_state(
            loss,
            sample_weight=valid_count,
        )

        self._update_masked_metrics(
            labels,
            predictions,
            valid_mask,
        )

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }


# =============================================================================
# Post-training validation / threshold selection
# =============================================================================

def confusion_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
):
    pred = probabilities >= threshold

    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )
    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )
    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )
    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn)
    )

    neg_precision = (
        tn / (tn + fn)
        if tn + fn > 0
        else 0.0
    )
    neg_recall = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )
    neg_f1 = (
        2.0 * neg_precision * neg_recall
        / (neg_precision + neg_recall)
        if neg_precision + neg_recall > 0
        else 0.0
    )
    macro_f1 = 0.5 * (f1 + neg_f1)

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
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def scan_validation_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):
    thresholds = np.round(
        np.arange(0.05, 0.9501, 0.005),
        3,
    )

    rows = [
        confusion_metrics(
            y_true,
            probabilities,
            float(threshold),
        )
        for threshold in thresholds
    ]

    best = max(
        rows,
        key=lambda row: (
            row["informal_f1"],
            row["macro_f1"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )

    return rows, best


def compute_pr_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    metric = tf.keras.metrics.AUC(
        curve="PR"
    )
    metric.update_state(
        y_true.astype(np.float32),
        probabilities.astype(np.float32),
    )
    return float(metric.result().numpy())


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    set_reproducibility(SEED)
    ensure_clean_start()

    print("=" * 80)
    print("Nairobi v2 — revised S2-only model training")
    print("=" * 80)
    print("Python:", sys.executable)
    print("TensorFlow:", tf.__version__)
    print("Seed:", SEED)
    print("Batch size:", BATCH_SIZE)
    print("Max epochs:", MAX_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Dropout:", DROPOUT_RATE)
    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    physical_gpus = tf.config.list_physical_devices(
        "GPU"
    )
    print("Physical GPUs:", physical_gpus)

    if not physical_gpus:
        raise RuntimeError(
            "TensorFlow detected no GPU inside the Slurm GPU job"
        )

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except RuntimeError as exc:
            print("Memory-growth note:", exc)

    logical_gpus = tf.config.list_logical_devices(
        "GPU"
    )
    print("Logical GPUs:", logical_gpus)

    (
        x_train,
        y_train_packed,
        x_val,
        y_val_packed,
        y_val_raw,
        mask_val,
        counts,
    ) = load_and_validate_data()

    positive_weight = (
        counts["train_negative"]
        / counts["train_positive"]
    )

    print(
        "Positive-class weight "
        "(train_negative/train_positive):",
        positive_weight,
    )
    print(
        "Loss normalization:",
        "sum(valid weighted BCE) / number of valid cells",
    )

    core_model, encoder = (
        build_core_model_and_encoder()
    )

    if core_model.output_shape != (
        None,
        16,
        16,
        1,
    ):
        raise ValueError(
            f"Unexpected model output shape: "
            f"{core_model.output_shape}"
        )

    if encoder.output_shape != (
        None,
        16,
        16,
        256,
    ):
        raise ValueError(
            f"Unexpected encoder output shape: "
            f"{encoder.output_shape}"
        )

    training_model = MaskedS2TrainingModel(
        core_model=core_model,
        positive_weight=positive_weight,
        name="nairobi_v2_masked_s2_training",
    )

    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
    )

    # Build wrapper before summary/checkpoint.
    _ = training_model(
        tf.zeros(
            [1, 128, 128, 4],
            dtype=tf.float32,
        ),
        training=False,
    )

    core_model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_WEIGHTS),
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
        tf.keras.callbacks.CSVLogger(
            str(HISTORY_CSV),
            append=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    started_at = datetime.now().isoformat()
    run_start = time.time()

    history = training_model.fit(
        x=x_train,
        y=y_train_packed,
        validation_data=(
            x_val,
            y_val_packed,
        ),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    elapsed_seconds = time.time() - run_start

    if not BEST_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not created: {BEST_WEIGHTS}"
        )

    # Explicitly reload the validation-selected checkpoint.
    training_model.load_weights(BEST_WEIGHTS)

    val_probs = core_model.predict(
        x_val,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).astype(np.float32, copy=False)

    expected_val_prob_shape = (
        EXPECTED["val_label_shape"]
        + (1,)
    )
    if val_probs.shape != expected_val_prob_shape:
        raise ValueError(
            f"Unexpected val probability shape: "
            f"{val_probs.shape}"
        )

    if not np.isfinite(val_probs).all():
        raise ValueError(
            "Validation probabilities contain non-finite values"
        )

    if (
        float(val_probs.min()) < 0.0
        or float(val_probs.max()) > 1.0
    ):
        raise ValueError(
            "Validation probabilities outside [0,1]"
        )

    np.save(
        VAL_PROBS_NPY,
        val_probs,
        allow_pickle=False,
    )

    val_true_valid = (
        y_val_raw[mask_val]
        .astype(np.int8, copy=False)
    )
    val_prob_valid = (
        val_probs[..., 0][mask_val]
        .astype(np.float32, copy=False)
    )

    threshold_rows, best_threshold_row = (
        scan_validation_thresholds(
            val_true_valid,
            val_prob_valid,
        )
    )

    with THRESHOLD_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = list(
            threshold_rows[0].keys()
        )
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(threshold_rows)

    val_at_05 = confusion_metrics(
        val_true_valid,
        val_prob_valid,
        0.5,
    )
    val_pr_auc = compute_pr_auc(
        val_true_valid,
        val_prob_valid,
    )
    val_brier = float(
        np.mean(
            (
                val_prob_valid
                - val_true_valid.astype(np.float32)
            )
            ** 2
        )
    )

    # The encoder and core model share trained layer objects. At this point
    # the wrapper contains the validation-selected checkpoint.
    core_model.save(FULL_MODEL_PATH)
    encoder.save(ENCODER_PATH)

    completed_at = datetime.now().isoformat()
    epochs_ran = len(history.history["loss"])
    best_epoch = int(
        np.argmax(history.history["val_pr_auc"])
    ) + 1

    config = {
        "created_at": completed_at,
        "seed": SEED,
        "python": sys.executable,
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "physical_gpus": [
            str(device)
            for device in physical_gpus
        ],
        "train_input": str(X_TRAIN_PATH),
        "val_input": str(X_VAL_PATH),
        "train_labels": str(Y_TRAIN_PATH),
        "val_labels": str(Y_VAL_PATH),
        "train_mask": str(MASK_TRAIN_PATH),
        "val_mask": str(MASK_VAL_PATH),
        "test_policy": (
            "Test arrays are not loaded or evaluated during training, "
            "early stopping, checkpoint selection, or threshold selection."
        ),
        "counts": counts,
        "positive_weight": positive_weight,
        "loss_definition": (
            "sum over valid cells of class-weighted BCE divided by "
            "the exact number of valid cells; invalid cells contribute "
            "neither numerator nor denominator"
        ),
        "architecture": {
            "input": [128, 128, 4],
            "encoder": [
                "Conv2D 64 x2 + MaxPool2D",
                "Conv2D 128 x2 + MaxPool2D",
                "Conv2D 256 x2 + MaxPool2D",
            ],
            "encoder_output": [16, 16, 256],
            "dropout": DROPOUT_RATE,
            "prediction_head": (
                "Dropout -> Conv2D(1, kernel_size=1, sigmoid)"
            ),
            "output": [16, 16, 1],
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "optimizer": "Adam",
            "initial_learning_rate": LEARNING_RATE,
            "checkpoint_monitor": "val_pr_auc",
            "checkpoint_mode": "max",
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "lr_reduce_patience": LR_PATIENCE,
            "lr_reduce_factor": 0.5,
            "min_learning_rate": 1e-6,
            "elapsed_seconds": elapsed_seconds,
        },
        "threshold_selection": {
            "split": "validation only",
            "criterion": (
                "maximize informal F1; tie-break by macro-F1, "
                "distance to 0.5, then lower threshold"
            ),
            "scan_start": 0.05,
            "scan_stop": 0.95,
            "scan_step": 0.005,
        },
    }

    atomic_write_json(CONFIG_JSON, config)

    audit = {
        "created_at": completed_at,
        "status": "PASSED",
        "seed": SEED,
        "epochs_ran": epochs_ran,
        "best_epoch_by_val_pr_auc": best_epoch,
        "val_pr_auc": val_pr_auc,
        "val_brier": val_brier,
        "val_metrics_threshold_0_5": val_at_05,
        "selected_validation_threshold": (
            best_threshold_row
        ),
        "model_path": str(FULL_MODEL_PATH),
        "encoder_path": str(ENCODER_PATH),
        "best_weights": str(BEST_WEIGHTS),
        "val_probability_path": str(
            VAL_PROBS_NPY
        ),
        "history_path": str(HISTORY_CSV),
        "threshold_scan_path": str(
            THRESHOLD_CSV
        ),
        "encoder_output_shape": [
            16,
            16,
            256,
        ],
        "loss_valid_cell_normalization": True,
        "test_data_loaded": False,
        "test_data_evaluated": False,
    }

    atomic_write_json(AUDIT_JSON, audit)

    audit_lines = [
        "=" * 80,
        "Nairobi v2 — Revised S2-only Training Audit",
        "=" * 80,
        f"Seed: {SEED}",
        f"Train valid cells: {counts['train_valid']}",
        (
            "Train class counts: "
            f"negative={counts['train_negative']}, "
            f"positive={counts['train_positive']}"
        ),
        f"Validation valid cells: {counts['val_valid']}",
        (
            "Validation class counts: "
            f"negative={counts['val_negative']}, "
            f"positive={counts['val_positive']}"
        ),
        f"Positive-class weight: {positive_weight}",
        (
            "Loss normalization: EXACT VALID-CELL MEAN "
            "(invalid cells excluded from numerator and denominator)"
        ),
        "Model input: 128x128x4",
        "Encoder output: 16x16x256",
        "Prediction head: Dropout -> 1x1 Conv sigmoid",
        "Prediction output: 16x16x1",
        f"Epochs run: {epochs_ran}",
        f"Best epoch by validation PR-AUC: {best_epoch}",
        f"Validation PR-AUC: {val_pr_auc}",
        f"Validation Brier score: {val_brier}",
        (
            "Validation @0.5 informal F1: "
            f"{val_at_05['informal_f1']}"
        ),
        (
            "Selected validation threshold: "
            f"{best_threshold_row['threshold']}"
        ),
        (
            "Selected-threshold informal precision: "
            f"{best_threshold_row['informal_precision']}"
        ),
        (
            "Selected-threshold informal recall: "
            f"{best_threshold_row['informal_recall']}"
        ),
        (
            "Selected-threshold informal F1: "
            f"{best_threshold_row['informal_f1']}"
        ),
        (
            "Selected-threshold informal IoU: "
            f"{best_threshold_row['informal_iou']}"
        ),
        (
            "Selected-threshold macro-F1: "
            f"{best_threshold_row['macro_f1']}"
        ),
        "TEST DATA LOADED: NO",
        "TEST DATA EVALUATED: NO",
        "",
        "S2-ONLY TRAINING: PASSED",
    ]

    AUDIT_TXT.write_text(
        "\n".join(audit_lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(audit_lines))
    print()
    print("Saved full model:", FULL_MODEL_PATH)
    print("Saved encoder:", ENCODER_PATH)
    print("Saved history:", HISTORY_CSV)
    print("Saved threshold scan:", THRESHOLD_CSV)
    print(
        "Saved validation probabilities:",
        VAL_PROBS_NPY,
    )
    print("Saved config:", CONFIG_JSON)
    print("Saved audit JSON:", AUDIT_JSON)
    print("Saved audit TXT:", AUDIT_TXT)


if __name__ == "__main__":
    main()
