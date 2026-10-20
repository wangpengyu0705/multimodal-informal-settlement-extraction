#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import time
from typing import Iterable

import numpy as np
import tensorflow as tf


# =============================================================================
# Fixed project settings
# =============================================================================

PROJECT = Path.home() / "projects" / "nairobi_v2"
METADATA_ROOT = PROJECT / "metadata"
GSV_ROOT = PROJECT / "scratch/data/gsv/images"

OUT = (
    PROJECT
    / "scratch/features/gsv_embeddings/resnet50_imagenet_v2"
)
OUT.mkdir(parents=True, exist_ok=True)

FINAL_EMBEDDINGS = OUT / "gsv_resnet50_embeddings_v2.npy"
PARTIAL_EMBEDDINGS = OUT / "gsv_resnet50_embeddings_v2.partial.npy"
INDEX_CSV = OUT / "gsv_resnet50_embedding_index_v2.csv"
PROGRESS_JSON = OUT / "gsv_resnet50_embedding_progress_v2.json"
CONFIG_JSON = OUT / "gsv_resnet50_embedding_config_v2.json"
AUDIT_JSON = OUT / "gsv_resnet50_embedding_audit_v2.json"
AUDIT_TXT = OUT / "gsv_resnet50_embedding_audit_v2.txt"

EXPECTED_IMAGES = 34207
EMBEDDING_DIM = 2048
TARGET_HEIGHT = 128
TARGET_WIDTH = 500
BATCH_SIZE = int(os.environ.get("GSV_BATCH_SIZE", "32"))
PROGRESS_EVERY_BATCHES = int(
    os.environ.get("GSV_PROGRESS_EVERY_BATCHES", "10")
)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MANIFEST_PRIORITY = [
    METADATA_ROOT / "full_gsv_manifest_aligned_v2_portable.csv",
    METADATA_ROOT
    / "latest_full_gsv_aligned_v2/full_gsv_manifest_aligned_v2.csv",
    METADATA_ROOT / "full_gsv_manifest_aligned_v2.csv",
]

PATH_COLUMN_PRIORITY = [
    "portable_relative_path",
    "hpc_relative_path",
    "target_relative_path",
    "image_relative_path",
    "relative_path",
    "target_path",
    "image_path",
    "source_path",
]


# =============================================================================
# Utilities
# =============================================================================

def atomic_write_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def discover_manifest() -> Path:
    candidates: list[Path] = []

    for path in MANIFEST_PRIORITY:
        if path.is_file() and path not in candidates:
            candidates.append(path)

    patterns = [
        "**/*gsv*manifest*aligned*v2*.csv",
        "**/*full*gsv*manifest*.csv",
        "**/*gsv*manifest*.csv",
    ]

    for pattern in patterns:
        for path in sorted(METADATA_ROOT.glob(pattern)):
            if path.is_file() and path not in candidates:
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "No GSV manifest CSV was found under "
            f"{METADATA_ROOT}"
        )

    print("Manifest candidates:")
    for rank, path in enumerate(candidates, start=1):
        print(f"  {rank}. {path}")

    # The first candidate follows the explicit priority list.
    return candidates[0]


def read_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"Manifest has no header: {path}")

    if not rows:
        raise ValueError(f"Manifest has no data rows: {path}")

    return fieldnames, rows


def normalize_path_value(value: str) -> str:
    return value.strip().replace("\\", "/")


def resolve_candidate_path(value: str) -> Path | None:
    value = normalize_path_value(value)
    if not value:
        return None

    candidate = Path(value)

    if candidate.is_absolute():
        if candidate.is_file():
            return candidate.resolve()
        return None

    # Common portable manifests may include an "images/" prefix,
    # while GSV_ROOT itself already points to that directory.
    relative_options = [candidate]

    parts = candidate.parts
    if parts and parts[0].lower() == "images":
        relative_options.append(Path(*parts[1:]))

    for relative_path in relative_options:
        resolved = GSV_ROOT / relative_path
        if resolved.is_file():
            return resolved.resolve()

    return None


def choose_path_column(fieldnames: list[str], rows: list[dict]):
    candidate_columns = [
        column
        for column in PATH_COLUMN_PRIORITY
        if column in fieldnames
    ]

    # Also consider any unanticipated path-like columns.
    for column in fieldnames:
        lower = column.lower()
        if (
            "path" in lower
            and column not in candidate_columns
        ):
            candidate_columns.append(column)

    if not candidate_columns:
        raise ValueError(
            "No path-like column was found in the manifest. "
            f"Columns: {fieldnames}"
        )

    evaluations = []

    for column in candidate_columns:
        resolved_paths: list[Path | None] = []
        existing_count = 0

        for row in rows:
            resolved = resolve_candidate_path(row.get(column, ""))
            resolved_paths.append(resolved)
            if resolved is not None:
                existing_count += 1

        evaluations.append(
            {
                "column": column,
                "existing_count": existing_count,
                "resolved_paths": resolved_paths,
            }
        )

    evaluations.sort(
        key=lambda item: (
            item["existing_count"],
            -candidate_columns.index(item["column"]),
        ),
        reverse=True,
    )

    print("Manifest path-column evaluation:")
    for item in evaluations:
        print(
            f"  {item['column']}: "
            f"{item['existing_count']:,} existing files"
        )

    best = evaluations[0]

    if best["existing_count"] != EXPECTED_IMAGES:
        raise ValueError(
            "Could not resolve exactly "
            f"{EXPECTED_IMAGES:,} manifest images. "
            f"Best column {best['column']!r} resolved "
            f"{best['existing_count']:,}."
        )

    return best["column"], best["resolved_paths"]


def enumerate_actual_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"GSV image root not found: {root}")

    images = sorted(
        path.resolve()
        for path in root.rglob("*")
        if (
            path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() in VALID_EXTENSIONS
        )
    )

    return images


def prepare_manifest_index():
    manifest_path = discover_manifest()
    fieldnames, rows = read_manifest(manifest_path)

    print("Selected manifest:", manifest_path)
    print("Manifest rows:", f"{len(rows):,}")
    print("Manifest columns:", fieldnames)

    path_column, resolved_paths = choose_path_column(
        fieldnames,
        rows,
    )

    selected_rows = []
    selected_paths = []

    for manifest_row_number, (row, resolved) in enumerate(
        zip(rows, resolved_paths),
        start=1,
    ):
        if resolved is None:
            continue

        selected_rows.append(
            {
                **row,
                "_manifest_row_number": manifest_row_number,
            }
        )
        selected_paths.append(resolved)

    if len(selected_rows) != EXPECTED_IMAGES:
        raise ValueError(
            f"Selected manifest rows {len(selected_rows):,} "
            f"!= {EXPECTED_IMAGES:,}"
        )

    path_strings = [str(path) for path in selected_paths]

    if len(set(path_strings)) != EXPECTED_IMAGES:
        raise ValueError("Duplicate resolved image paths in manifest")

    actual_images = enumerate_actual_images(GSV_ROOT)

    print("Actual image files under GSV_ROOT:", f"{len(actual_images):,}")

    if len(actual_images) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES:,} actual image files, "
            f"found {len(actual_images):,}"
        )

    actual_set = {str(path) for path in actual_images}
    selected_set = set(path_strings)

    missing_from_manifest = actual_set - selected_set
    missing_on_disk = selected_set - actual_set

    if missing_from_manifest or missing_on_disk:
        raise ValueError(
            "Manifest/image set mismatch: "
            f"actual_not_in_manifest={len(missing_from_manifest)}, "
            f"manifest_not_on_disk={len(missing_on_disk)}"
        )

    print("Manifest ↔ extracted image set equality: PASSED")
    print("Selected path column:", path_column)
    print("Embedding row order: selected manifest row order")

    return (
        manifest_path,
        fieldnames,
        selected_rows,
        selected_paths,
        path_column,
    )


def write_embedding_index(
    fieldnames: list[str],
    selected_rows: list[dict],
    selected_paths: list[Path],
    path_column: str,
) -> None:
    output_fields = [
        "embedding_row",
        "manifest_row_number",
        "resolved_image_path",
        "resolved_relative_path",
        "manifest_path_column",
    ] + fieldnames

    with INDEX_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=output_fields,
            extrasaction="ignore",
        )
        writer.writeheader()

        for embedding_row, (row, path) in enumerate(
            zip(selected_rows, selected_paths)
        ):
            try:
                relative_path = path.relative_to(GSV_ROOT.resolve())
            except ValueError:
                relative_path = path

            output_row = {
                "embedding_row": embedding_row,
                "manifest_row_number": row["_manifest_row_number"],
                "resolved_image_path": str(path),
                "resolved_relative_path": str(relative_path),
                "manifest_path_column": path_column,
            }

            for field in fieldnames:
                output_row[field] = row.get(field, "")

            writer.writerow(output_row)


# =============================================================================
# TensorFlow input pipeline
# =============================================================================

def decode_resize_preprocess(path_tensor: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path_tensor)

    image = tf.io.decode_image(
        raw,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [TARGET_HEIGHT, TARGET_WIDTH],
        method="bilinear",
        antialias=True,
    )

    image = tf.cast(image, tf.float32)

    return tf.keras.applications.resnet50.preprocess_input(
        image
    )


def build_dataset(paths: list[str], start_row: int):
    remaining_paths = paths[start_row:]

    dataset = tf.data.Dataset.from_tensor_slices(
        remaining_paths
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    dataset = dataset.map(
        decode_resize_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =============================================================================
# Resume/output handling
# =============================================================================

def load_resume_row() -> int:
    final_exists = FINAL_EMBEDDINGS.exists()
    partial_exists = PARTIAL_EMBEDDINGS.exists()
    progress_exists = PROGRESS_JSON.exists()

    if final_exists:
        raise FileExistsError(
            "Final embeddings already exist. Refusing to overwrite: "
            f"{FINAL_EMBEDDINGS}"
        )

    if partial_exists != progress_exists:
        raise RuntimeError(
            "Inconsistent resume state: partial embedding file and "
            "progress JSON must either both exist or both be absent."
        )

    if not partial_exists:
        return 0

    progress = json.loads(
        PROGRESS_JSON.read_text(encoding="utf-8")
    )

    next_row = int(progress["next_row"])

    if not 0 <= next_row <= EXPECTED_IMAGES:
        raise ValueError(
            f"Invalid resume next_row: {next_row}"
        )

    return next_row


def open_embedding_memmap(start_row: int):
    if start_row == 0:
        return np.lib.format.open_memmap(
            PARTIAL_EMBEDDINGS,
            mode="w+",
            dtype=np.float32,
            shape=(EXPECTED_IMAGES, EMBEDDING_DIM),
        )

    mmap = np.lib.format.open_memmap(
        PARTIAL_EMBEDDINGS,
        mode="r+",
    )

    if mmap.shape != (EXPECTED_IMAGES, EMBEDDING_DIM):
        raise ValueError(
            f"Partial embedding shape is invalid: {mmap.shape}"
        )

    if mmap.dtype != np.float32:
        raise ValueError(
            f"Partial embedding dtype is invalid: {mmap.dtype}"
        )

    return mmap


def update_progress(
    next_row: int,
    started_at: str,
    batches_completed: int,
) -> None:
    atomic_write_json(
        PROGRESS_JSON,
        {
            "updated_at": datetime.now().isoformat(),
            "started_at": started_at,
            "status": "IN_PROGRESS",
            "next_row": next_row,
            "expected_rows": EXPECTED_IMAGES,
            "embedding_dimension": EMBEDDING_DIM,
            "batch_size": BATCH_SIZE,
            "batches_completed_this_run": batches_completed,
            "partial_file": str(PARTIAL_EMBEDDINGS),
        },
    )


def chunked_embedding_stats(path: Path):
    mmap = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if mmap.shape != (EXPECTED_IMAGES, EMBEDDING_DIM):
        raise ValueError(
            f"Final embedding shape is invalid: {mmap.shape}"
        )

    if mmap.dtype != np.float32:
        raise ValueError(
            f"Final embedding dtype is invalid: {mmap.dtype}"
        )

    chunk_rows = 256
    total_values = 0
    running_sum = 0.0
    running_sum_sq = 0.0
    global_min = np.inf
    global_max = -np.inf
    nonfinite_count = 0
    zero_vector_count = 0

    for start in range(0, EXPECTED_IMAGES, chunk_rows):
        stop = min(start + chunk_rows, EXPECTED_IMAGES)
        chunk = np.asarray(mmap[start:stop])

        finite_mask = np.isfinite(chunk)
        nonfinite_count += int((~finite_mask).sum())

        if finite_mask.any():
            values = chunk[finite_mask].astype(
                np.float64,
                copy=False,
            )
            global_min = min(
                global_min,
                float(values.min()),
            )
            global_max = max(
                global_max,
                float(values.max()),
            )
            running_sum += float(values.sum())
            running_sum_sq += float(np.square(values).sum())
            total_values += int(values.size)

        zero_vector_count += int(
            np.all(chunk == 0.0, axis=1).sum()
        )

    if nonfinite_count:
        raise ValueError(
            f"Final embeddings contain {nonfinite_count} "
            "non-finite values"
        )

    mean = running_sum / total_values
    variance = (
        running_sum_sq / total_values
        - mean * mean
    )
    variance = max(variance, 0.0)
    std = variance ** 0.5

    return {
        "shape": list(mmap.shape),
        "dtype": str(mmap.dtype),
        "min": global_min,
        "max": global_max,
        "mean": mean,
        "std": std,
        "nonfinite_count": nonfinite_count,
        "zero_vector_count": zero_vector_count,
        "bytes": int(path.stat().st_size),
    }


# =============================================================================
# Main extraction
# =============================================================================

def main() -> None:
    if BATCH_SIZE <= 0:
        raise ValueError(
            f"GSV_BATCH_SIZE must be positive, got {BATCH_SIZE}"
        )

    print("=" * 80)
    print("Nairobi v2 — batch GSV ResNet50 embedding extraction")
    print("=" * 80)
    print("Python:", sys.executable)
    print("TensorFlow:", tf.__version__)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Batch size:", BATCH_SIZE)
    print("GSV root:", GSV_ROOT)
    print("Output:", OUT)

    physical_gpus = tf.config.list_physical_devices("GPU")
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

    logical_gpus = tf.config.list_logical_devices("GPU")
    print("Logical GPUs:", logical_gpus)

    if not logical_gpus:
        raise RuntimeError(
            "TensorFlow reported no logical GPU"
        )

    (
        manifest_path,
        manifest_fields,
        selected_rows,
        selected_paths,
        path_column,
    ) = prepare_manifest_index()

    write_embedding_index(
        manifest_fields,
        selected_rows,
        selected_paths,
        path_column,
    )

    print("Saved embedding index:", INDEX_CSV)

    path_strings = [str(path) for path in selected_paths]

    started_at = datetime.now().isoformat()
    resume_row = load_resume_row()

    print("Resume row:", resume_row)

    embeddings = open_embedding_memmap(resume_row)

    model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 3),
    )
    model.trainable = False

    # Warm-up before starting the formal write loop.
    warmup_input = tf.zeros(
        [1, TARGET_HEIGHT, TARGET_WIDTH, 3],
        dtype=tf.float32,
    )
    with tf.device("/GPU:0"):
        warmup_output = model(
            warmup_input,
            training=False,
        )

    if tuple(warmup_output.shape) != (1, EMBEDDING_DIM):
        raise ValueError(
            f"Warm-up output shape is invalid: "
            f"{tuple(warmup_output.shape)}"
        )

    if "GPU" not in warmup_output.device.upper():
        raise RuntimeError(
            f"Warm-up output was not placed on GPU: "
            f"{warmup_output.device}"
        )

    print("Warm-up output device:", warmup_output.device)
    print("Warm-up: PASSED")

    dataset = build_dataset(
        path_strings,
        resume_row,
    )

    current_row = resume_row
    batches_completed = 0
    run_start_time = time.time()

    update_progress(
        current_row,
        started_at,
        batches_completed,
    )

    for batch in dataset:
        with tf.device("/GPU:0"):
            batch_embeddings_tensor = model(
                batch,
                training=False,
            )

        batch_embeddings = (
            batch_embeddings_tensor
            .numpy()
            .astype(np.float32, copy=False)
        )

        batch_n = batch_embeddings.shape[0]

        if batch_embeddings.shape != (
            batch_n,
            EMBEDDING_DIM,
        ):
            raise ValueError(
                "Unexpected batch embedding shape: "
                f"{batch_embeddings.shape}"
            )

        if not np.isfinite(batch_embeddings).all():
            raise ValueError(
                f"Non-finite embedding values in rows "
                f"{current_row}:{current_row + batch_n}"
            )

        next_row = current_row + batch_n
        embeddings[current_row:next_row] = batch_embeddings
        current_row = next_row
        batches_completed += 1

        if (
            batches_completed % PROGRESS_EVERY_BATCHES == 0
            or current_row == EXPECTED_IMAGES
        ):
            embeddings.flush()
            update_progress(
                current_row,
                started_at,
                batches_completed,
            )

            elapsed = time.time() - run_start_time
            processed_this_run = current_row - resume_row
            rate = (
                processed_this_run / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"Progress: {current_row:,}/{EXPECTED_IMAGES:,} "
                f"({100.0 * current_row / EXPECTED_IMAGES:.2f}%), "
                f"this_run_rate={rate:.2f} images/s",
                flush=True,
            )

    embeddings.flush()
    del embeddings

    if current_row != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Extraction stopped at row {current_row:,}; "
            f"expected {EXPECTED_IMAGES:,}"
        )

    PARTIAL_EMBEDDINGS.replace(FINAL_EMBEDDINGS)

    stats = chunked_embedding_stats(FINAL_EMBEDDINGS)
    sha256 = sha256_file(FINAL_EMBEDDINGS)

    completed_at = datetime.now().isoformat()
    elapsed_seconds = time.time() - run_start_time

    config = {
        "created_at": completed_at,
        "python": sys.executable,
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "physical_gpus": [
            str(device)
            for device in physical_gpus
        ],
        "logical_gpus": [
            str(device)
            for device in logical_gpus
        ],
        "manifest": str(manifest_path),
        "manifest_path_column": path_column,
        "gsv_root": str(GSV_ROOT),
        "expected_images": EXPECTED_IMAGES,
        "target_height": TARGET_HEIGHT,
        "target_width": TARGET_WIDTH,
        "batch_size": BATCH_SIZE,
        "model": {
            "architecture": "ResNet50",
            "weights": "imagenet",
            "include_top": False,
            "pooling": "avg",
            "trainable": False,
            "output_dimension": EMBEDDING_DIM,
        },
        "preprocessing": (
            "tf.io.decode_image -> resize 128x500 float32 -> "
            "tf.keras.applications.resnet50.preprocess_input"
        ),
        "embedding_order": (
            "selected manifest row order, filtered to the "
            "34207 images present under GSV_ROOT"
        ),
        "resume_row_at_start": resume_row,
        "elapsed_seconds_this_run": elapsed_seconds,
    }

    atomic_write_json(CONFIG_JSON, config)

    audit = {
        "created_at": completed_at,
        "status": "PASSED",
        "manifest_image_count": len(selected_paths),
        "actual_image_count": EXPECTED_IMAGES,
        "manifest_actual_set_equality": True,
        "embedding_index": str(INDEX_CSV),
        "embeddings": str(FINAL_EMBEDDINGS),
        "embedding_statistics": stats,
        "sha256": sha256,
        "model_output_device_warmup": warmup_output.device,
        "all_values_finite": stats["nonfinite_count"] == 0,
        "zero_vector_count": stats["zero_vector_count"],
    }

    atomic_write_json(AUDIT_JSON, audit)

    audit_lines = [
        "=" * 80,
        "Nairobi v2 — GSV ResNet50 Embedding Extraction Audit",
        "=" * 80,
        f"Manifest: {manifest_path}",
        f"Manifest path column: {path_column}",
        f"GSV root: {GSV_ROOT}",
        f"Images: {EXPECTED_IMAGES}",
        "Manifest ↔ extracted image set equality: PASSED",
        "Embedding order: selected manifest row order",
        "Model: frozen ImageNet ResNet50",
        "Preprocessing: official keras ResNet50 preprocess_input",
        f"Input shape: {TARGET_HEIGHT}x{TARGET_WIDTH}x3",
        f"Embedding shape: {stats['shape']}",
        f"Embedding dtype: {stats['dtype']}",
        f"Embedding min: {stats['min']}",
        f"Embedding max: {stats['max']}",
        f"Embedding mean: {stats['mean']}",
        f"Embedding std: {stats['std']}",
        f"Non-finite values: {stats['nonfinite_count']}",
        f"Zero vectors: {stats['zero_vector_count']}",
        f"SHA256: {sha256}",
        f"Warm-up device: {warmup_output.device}",
        "",
        "GSV RESNET50 EMBEDDING EXTRACTION: PASSED",
    ]

    AUDIT_TXT.write_text(
        "\n".join(audit_lines),
        encoding="utf-8",
    )

    atomic_write_json(
        PROGRESS_JSON,
        {
            "updated_at": completed_at,
            "started_at": started_at,
            "status": "COMPLETED",
            "next_row": EXPECTED_IMAGES,
            "expected_rows": EXPECTED_IMAGES,
            "embedding_dimension": EMBEDDING_DIM,
            "batch_size": BATCH_SIZE,
            "final_file": str(FINAL_EMBEDDINGS),
            "sha256": sha256,
        },
    )

    print()
    print("\n".join(audit_lines))
    print()
    print("Saved embeddings:", FINAL_EMBEDDINGS)
    print("Saved index:", INDEX_CSV)
    print("Saved config:", CONFIG_JSON)
    print("Saved audit JSON:", AUDIT_JSON)
    print("Saved audit TXT:", AUDIT_TXT)
    print("Saved progress:", PROGRESS_JSON)


if __name__ == "__main__":
    main()
