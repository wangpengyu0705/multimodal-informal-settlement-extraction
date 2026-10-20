#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import tensorflow as tf


PROJECT = Path.home() / "projects" / "nairobi_v2"
SEED = 42
K = 7
BASE_DIM = 2048
FEATURE_DIM = 256
N_COVERED = 5744
FUSION_THRESHOLD = 0.59
EPS = 1e-7
FULL_RECON_TOL = 1e-4

# Frozen protocol / final-test / Step-16 outputs.
FREEZE_MANIFEST = (
    PROJECT / "results/final_protocol_v2/seed_42/final_protocol_manifest_v2.json"
)
FINAL_TEST_AUDIT = (
    PROJECT / "results/final_test_v2/seed_42/final_test_audit_v2.json"
)
COVERED_CELL_INDEX = (
    PROJECT / "results/final_test_v2/seed_42/covered_test_cell_index_v2.csv"
)
FROZEN_FUSION_PROBS = (
    PROJECT
    / "results/final_test_v2/seed_42/"
      "fusion_mean_test_probabilities_covered_v2.npy"
)
STEP16_CELLWISE = (
    PROJECT / "results/xai_modality_v2/seed_42/xai_modality_delta_cellwise_v2.csv"
)

# Frozen model/data artifacts.
X_TEST_S2 = PROJECT / "scratch/features/s2_inputs/X_test_s2_v2.npy"
S2_ENCODER = PROJECT / "models/s2_only_v2/seed_42/s2_encoder_best_v2.keras"
GSV_ENCODER = (
    PROJECT / "models/gsv_mean_pool_v2/seed_42/gsv_encoder_mean_pool_best_v2.keras"
)
FUSION_MODEL = (
    PROJECT / "models/fusion_mean_pool_v2/seed_42/fusion_mean_pool_best_v2.keras"
)
FUSION_SCALER = (
    PROJECT
    / "results/fusion_mean_pool_v2/seed_42/"
      "fusion_mean_pool_standardization_train_only_v2.npz"
)
GSV_EMBEDDINGS = (
    PROJECT
    / "scratch/features/gsv_embeddings/resnet50_imagenet_v2/"
      "gsv_resnet50_embeddings_v2.npy"
)
GSV_EMBED_INDEX = (
    PROJECT
    / "scratch/features/gsv_embeddings/resnet50_imagenet_v2/"
      "gsv_resnet50_embedding_index_v2.csv"
)

# Outputs.
OUT_DIR = PROJECT / "results/xai_loo_v2/seed_42"
IMAGEWISE_CSV = OUT_DIR / "loo_imagewise_influence_v2.csv"
CELLWISE_CSV = OUT_DIR / "loo_cellwise_summary_v2.csv"
GROUP_SUMMARY_CSV = OUT_DIR / "loo_group_summary_v2.csv"
K_SUMMARY_CSV = OUT_DIR / "loo_k_summary_v2.csv"
AUDIT_JSON = OUT_DIR / "loo_audit_v2.json"
AUDIT_TXT = OUT_DIR / "loo_audit_v2.txt"


@tf.keras.utils.register_keras_serializable(package="NairobiV2")
class MaskedMeanPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        z, slot_mask = inputs
        mask = tf.cast(slot_mask, z.dtype)[..., None]
        numerator = tf.reduce_sum(z * mask, axis=1)
        denominator = tf.reduce_sum(mask, axis=1)
        return tf.math.divide_no_nan(numerator, denominator)


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


def parse_int(row: dict, key: str) -> int:
    return int(float(str(row[key]).strip()))


def parse_float(row: dict, key: str) -> float:
    return float(str(row[key]).strip())


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def ensure_clean_start():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protected = [
        IMAGEWISE_CSV,
        CELLWISE_CSV,
        GROUP_SUMMARY_CSV,
        K_SUMMARY_CSV,
        AUDIT_JSON,
        AUDIT_TXT,
    ]
    existing = [p for p in protected if p.exists()]
    if existing:
        raise FileExistsError(
            "Step-17 LOO outputs already exist; refusing overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )


def verify_frozen_state():
    protocol = load_json(FREEZE_MANIFEST)
    final_test = load_json(FINAL_TEST_AUDIT)

    if protocol.get("status") != "FROZEN":
        raise ValueError("Protocol is not FROZEN")
    if final_test.get("status") != "PASSED":
        raise ValueError("Final test audit is not PASSED")
    if int(protocol.get("primary_seed", -1)) != SEED:
        raise ValueError("Frozen primary seed is not 42")
    if int(final_test.get("primary_seed", -1)) != SEED:
        raise ValueError("Final-test primary seed is not 42")
    if final_test.get("post_test_retraining_or_tuning_allowed") is not False:
        raise ValueError("Final-test audit does not prohibit post-test tuning")

    frozen_hashes = protocol["frozen_artifact_hashes"]
    path_map = {
        "s2_encoder": S2_ENCODER,
        "gsv_mean_encoder": GSV_ENCODER,
        "fusion_mean_model": FUSION_MODEL,
        "fusion_train_only_scaler": FUSION_SCALER,
    }
    for name, path in path_map.items():
        rec = frozen_hashes[name]
        if Path(rec["path"]) != path:
            raise ValueError(f"Frozen path mismatch: {name}")
        if sha256_file(path) != rec["sha256"]:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")

    threshold = float(protocol["final_models"]["fusion_mean"]["threshold"])
    if threshold != FUSION_THRESHOLD:
        raise ValueError(f"Fusion threshold changed: {threshold}")

    return protocol, final_test


def align_cell_metadata():
    idx_rows = read_csv_rows(COVERED_CELL_INDEX)
    xai_rows = read_csv_rows(STEP16_CELLWISE)

    if len(idx_rows) != N_COVERED or len(xai_rows) != N_COVERED:
        raise ValueError(
            f"Covered rows must be {N_COVERED}: "
            f"index={len(idx_rows)}, step16={len(xai_rows)}"
        )

    idx_by_order = {parse_int(r, "covered_order"): r for r in idx_rows}
    xai_by_order = {parse_int(r, "covered_order"): r for r in xai_rows}
    expected = set(range(N_COVERED))
    if set(idx_by_order) != expected or set(xai_by_order) != expected:
        raise ValueError("covered_order is not exact 0..5743")

    aligned = []
    for order in range(N_COVERED):
        idx = idx_by_order[order]
        xai = xai_by_order[order]
        for field in [
            "patch_order",
            "cell_row",
            "cell_col",
            "global_cell_row",
            "global_cell_col",
            "label",
            "slot_count",
        ]:
            if parse_int(idx, field) != parse_int(xai, field):
                raise ValueError(
                    f"Step16/index mismatch at order={order}, field={field}"
                )
        aligned.append((idx, xai))

    return aligned


def load_embedding_metadata():
    rows = read_csv_rows(GSV_EMBED_INDEX)
    meta = {}
    for r in rows:
        emb = parse_int(r, "embedding_row")
        if emb in meta:
            raise ValueError(f"Duplicate embedding_row={emb}")
        meta[emb] = {
            "fid_numeric": r.get("fid_numeric", ""),
            "filename": r.get("filename", ""),
            "relative_path": (
                r.get("resolved_relative_path")
                or r.get("image_path_relative_to_source")
                or r.get("resolved_image_path")
                or ""
            ),
        }
    if len(meta) != 34207:
        raise ValueError(f"Embedding metadata rows {len(meta)} != 34207")
    return meta


def build_all_records(aligned):
    records = []

    for order, (idx, xai) in enumerate(aligned):
        k = parse_int(idx, "slot_count")
        slot_rows = [
            parse_int(idx, f"slot_{s}_embedding_row")
            for s in range(1, K + 1)
        ]

        active = [e for e in slot_rows if e >= 0]
        if len(active) != k:
            raise ValueError(f"slot_count mismatch at covered_order={order}")
        if not 1 <= k <= K:
            raise ValueError(f"Invalid K={k} at covered_order={order}")

        records.append(
            {
                "covered_order": order,
                "patch_order": parse_int(idx, "patch_order"),
                "cell_row": parse_int(idx, "cell_row"),
                "cell_col": parse_int(idx, "cell_col"),
                "global_cell_row": parse_int(idx, "global_cell_row"),
                "global_cell_col": parse_int(idx, "global_cell_col"),
                "label": parse_int(idx, "label"),
                "slot_count": k,
                "slot_rows": slot_rows,
                "final_fusion_probability_saved": parse_float(
                    xai, "fusion_probability"
                ),
                "final_fusion_prediction_saved": parse_int(
                    xai, "fusion_prediction"
                ),
                "delta_gsv_logit": parse_float(xai, "delta_gsv_logit"),
                "prediction_change_group": xai["prediction_change_group"],
                "prediction_change_subtype": xai["prediction_change_subtype"],
            }
        )

    if len(records) != N_COVERED:
        raise ValueError(f"All covered records {len(records)} != {N_COVERED}")

    return records

def extract_s2_features(records):
    x_test = np.load(X_TEST_S2, allow_pickle=False)
    if x_test.shape != (102, 128, 128, 4):
        raise ValueError(f"Unexpected S2 test input shape {x_test.shape}")
    if x_test.dtype != np.float32 or not np.isfinite(x_test).all():
        raise ValueError("Invalid S2 test input")

    encoder = tf.keras.models.load_model(S2_ENCODER, compile=False)
    grid = encoder.predict(x_test, batch_size=16, verbose=0).astype(np.float32)
    if grid.shape != (102, 16, 16, FEATURE_DIM):
        raise ValueError(f"Unexpected S2 encoder output {grid.shape}")

    out = np.zeros((len(records), FEATURE_DIM), dtype=np.float32)
    for i, rec in enumerate(records):
        out[i] = grid[
            rec["patch_order"],
            rec["cell_row"],
            rec["cell_col"],
        ]

    if not np.isfinite(out).all():
        raise ValueError("Non-finite S2 covered features")
    return out


def build_gsv_tensor(records, embeddings):
    n = len(records)
    x = np.zeros((n, K, BASE_DIM), dtype=np.float32)
    mask = np.zeros((n, K), dtype=np.float32)

    for i, rec in enumerate(records):
        for slot, emb_row in enumerate(rec["slot_rows"]):
            if emb_row < 0:
                continue
            if emb_row >= embeddings.shape[0]:
                raise ValueError(f"Invalid embedding_row={emb_row}")
            x[i, slot] = embeddings[emb_row]
            mask[i, slot] = 1.0

    if not np.isfinite(x).all():
        raise ValueError("Non-finite GSV input tensor")

    observed_counts = mask.sum(axis=1).astype(int)
    expected_counts = np.asarray(
        [r["slot_count"] for r in records], dtype=int
    )
    if not np.array_equal(observed_counts, expected_counts):
        raise ValueError("GSV tensor/mask slot counts do not match metadata")
    if not np.all(observed_counts >= 1):
        raise ValueError("Covered records must all have at least one real GSV image")

    return x, mask

def load_scaler():
    data = np.load(FUSION_SCALER, allow_pickle=False)
    required = {"s2_mean", "s2_std", "gsv_mean", "gsv_std"}
    if set(data.files) != required:
        raise ValueError(f"Unexpected scaler keys: {set(data.files)}")

    arrays = {k: data[k].astype(np.float32) for k in required}
    data.close()

    for name, arr in arrays.items():
        if arr.shape != (FEATURE_DIM,) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid scaler array {name}")
    return arrays


def summary_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "mean": "",
            "median": "",
            "mean_abs": "",
            "median_abs": "",
            "positive_percent": "",
            "negative_percent": "",
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "median_abs": float(np.median(np.abs(values))),
        "positive_percent": float(np.mean(values > 0) * 100.0),
        "negative_percent": float(np.mean(values < 0) * 100.0),
    }


def main():
    ensure_clean_start()

    print("=" * 96)
    print("Nairobi v2 — XAI Step 17: Leave-One-Image-Out (LOO) GSV influence")
    print("=" * 96)
    print("Seed: 42")
    print("Training: NO")
    print("Threshold tuning: NO")
    print("Model modification: NO")
    print()
    print("Corrected reconstruction logic:")
    print("  1) reconstruct all 5,744 covered final-test cells first")
    print("  2) compare with frozen final-test Fusion probabilities")
    print("  3) only then select K>=2 cells for image-level LOO")
    print(
        "Definition: impact_i = logit(full Fusion) - "
        "logit(Fusion with image i omitted and mean recomputed)"
    )
    print()

    verify_frozen_state()
    print("[1] Frozen protocol/final-test/model hashes: PASSED")

    aligned = align_cell_metadata()
    all_records = build_all_records(aligned)
    print("[2] Full 5,744-cell final-test/Step16 alignment: PASSED")

    # Read the frozen one-time final-test Fusion probabilities directly.
    frozen_probs = np.load(FROZEN_FUSION_PROBS, allow_pickle=False)
    if frozen_probs.shape != (N_COVERED,):
        raise ValueError(
            f"Unexpected frozen Fusion probability shape {frozen_probs.shape}"
        )
    frozen_probs = frozen_probs.astype(np.float32)
    if not np.isfinite(frozen_probs).all():
        raise ValueError("Frozen Fusion probabilities contain non-finite values")

    # Step16 must still point to the exact frozen final-test predictions.
    step16_probs = np.asarray(
        [r["final_fusion_probability_saved"] for r in all_records],
        dtype=np.float32,
    )
    step16_frozen_diff = np.abs(step16_probs - frozen_probs)
    if float(np.max(step16_frozen_diff)) > 1e-7:
        raise ValueError(
            "Step16 probabilities do not match frozen final-test probabilities: "
            f"max_abs_diff={float(np.max(step16_frozen_diff))}"
        )
    print("[3] Step16 vs frozen final-test probability identity: PASSED")

    embedding_meta = load_embedding_metadata()
    embeddings = np.load(
        GSV_EMBEDDINGS,
        allow_pickle=False,
        mmap_mode="r",
    )
    if embeddings.shape != (34207, BASE_DIM):
        raise ValueError(f"Unexpected embedding table shape {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise ValueError(f"Unexpected embedding dtype {embeddings.dtype}")

    # IMPORTANT: build the exact full covered population BEFORE K filtering.
    gsv_x_all, gsv_mask_all = build_gsv_tensor(all_records, embeddings)
    del embeddings

    s2_features_all = extract_s2_features(all_records)
    scaler = load_scaler()

    s2_z_all = (
        (s2_features_all - scaler["s2_mean"]) / scaler["s2_std"]
    ).astype(np.float32)
    if not np.isfinite(s2_z_all).all():
        raise ValueError("Non-finite standardized S2 features")

    gsv_encoder = tf.keras.models.load_model(
        GSV_ENCODER,
        custom_objects={"MaskedMeanPooling": MaskedMeanPooling},
        compile=False,
    )
    fusion_model = tf.keras.models.load_model(FUSION_MODEL, compile=False)

    # Reconstruct Fusion on ALL 5,744 cells in frozen covered_order.
    full_gsv_features_all = gsv_encoder.predict(
        {
            "gsv_embeddings_k7": gsv_x_all,
            "gsv_slot_mask": gsv_mask_all,
        },
        batch_size=128,
        verbose=0,
    ).astype(np.float32)

    full_gsv_z_all = (
        (full_gsv_features_all - scaler["gsv_mean"]) / scaler["gsv_std"]
    ).astype(np.float32)

    full_recon_probs_all = fusion_model.predict(
        {
            "s2_feature_256": s2_z_all,
            "gsv_feature_256": full_gsv_z_all,
        },
        batch_size=128,
        verbose=0,
    ).astype(np.float32)[:, 0]

    recon_diff = np.abs(full_recon_probs_all - frozen_probs)
    max_recon_diff = float(np.max(recon_diff))
    mean_recon_diff = float(np.mean(recon_diff))
    median_recon_diff = float(np.median(recon_diff))
    q99_recon_diff = float(np.quantile(recon_diff, 0.99))
    frozen_pred = frozen_probs >= FUSION_THRESHOLD
    reconstructed_pred = full_recon_probs_all >= FUSION_THRESHOLD
    recon_flip_count = int(np.sum(frozen_pred != reconstructed_pred))

    print("[4] Full 5,744-cell Fusion reconstruction diagnostics:")
    print(f"    Max |reconstructed - frozen|: {max_recon_diff:.10f}")
    print(f"    Mean |reconstructed - frozen|: {mean_recon_diff:.10f}")
    print(f"    Median |reconstructed - frozen|: {median_recon_diff:.10f}")
    print(f"    Q99 |reconstructed - frozen|: {q99_recon_diff:.10f}")
    print(f"    Threshold-0.59 prediction flips: {recon_flip_count}")

    # Keep the original strict guard for this corrected rerun.
    # If it still fails, the diagnostics above distinguish implementation mismatch
    # from a small cross-run numerical execution difference.
    if max_recon_diff > FULL_RECON_TOL:
        raise ValueError(
            "Corrected full-population reconstruction still exceeds strict "
            f"tolerance {FULL_RECON_TOL}: max_abs_diff={max_recon_diff}; "
            f"mean_abs_diff={mean_recon_diff}; threshold_flips={recon_flip_count}"
        )

    print("[5] Full 5,744-cell frozen Fusion reconstruction: PASSED")

    # Only NOW select cells where within-cell omission is mathematically defined.
    loo_indices = np.asarray(
        [i for i, r in enumerate(all_records) if r["slot_count"] >= 2],
        dtype=np.int64,
    )
    k1_count = int(sum(r["slot_count"] == 1 for r in all_records))
    records = [all_records[i] for i in loo_indices]
    n_cells = len(records)
    n_images = int(sum(r["slot_count"] for r in records))

    if k1_count + n_cells != N_COVERED:
        raise ValueError("K=1 plus K>=2 accounting mismatch")

    print("[6] LOO population selected after frozen reconstruction:")
    print("    Covered cells total:", N_COVERED)
    print("    K=1 cells (LOO undefined):", k1_count)
    print("    K>=2 cells analyzed:", n_cells)
    print("    Real images in K>=2 cells:", n_images)

    # Slice the same-run full reconstruction for LOO.
    # Thus P_full and P_without_i are generated in the same Step17 run.
    gsv_x = gsv_x_all[loo_indices]
    gsv_mask = gsv_mask_all[loo_indices]
    s2_z = s2_z_all[loo_indices]
    full_recon_probs = full_recon_probs_all[loo_indices]
    full_logits = logit(full_recon_probs)

    # Preallocate omission outputs by (cell_index, slot).
    omission_prob = np.full((n_cells, K), np.nan, dtype=np.float64)

    # Omit one REAL slot at a time. Mean pooling is recomputed over K-1 real images.
    for slot in range(K):
        selected = np.where(gsv_mask[:, slot] > 0.5)[0]
        if len(selected) == 0:
            continue

        # Every selected record has original K>=2; removing one leaves >=1 real image.
        loo_mask = gsv_mask[selected].copy()
        loo_mask[:, slot] = 0.0

        loo_features = gsv_encoder.predict(
            {
                "gsv_embeddings_k7": gsv_x[selected],
                "gsv_slot_mask": loo_mask,
            },
            batch_size=128,
            verbose=0,
        ).astype(np.float32)

        loo_gsv_z = (
            (loo_features - scaler["gsv_mean"]) / scaler["gsv_std"]
        ).astype(np.float32)

        probs = fusion_model.predict(
            {
                "s2_feature_256": s2_z[selected],
                "gsv_feature_256": loo_gsv_z,
            },
            batch_size=128,
            verbose=0,
        ).astype(np.float32)[:, 0]

        omission_prob[selected, slot] = probs

    imagewise_rows = []
    per_cell_image_rows = defaultdict(list)

    for i, rec in enumerate(records):
        for slot in range(rec["slot_count"]):
            emb_row = rec["slot_rows"][slot]
            p_without = float(omission_prob[i, slot])
            if not np.isfinite(p_without):
                raise ValueError(
                    f"Missing omission result at cell={i}, slot={slot}"
                )

            impact = float(full_logits[i] - logit(p_without))
            prob_shift = float(full_recon_probs[i] - p_without)
            pred_without = int(p_without >= FUSION_THRESHOLD)
            full_pred = int(rec["final_fusion_prediction_saved"])
            meta = embedding_meta[emb_row]

            row = {
                "covered_order": rec["covered_order"],
                "patch_order": rec["patch_order"],
                "cell_row": rec["cell_row"],
                "cell_col": rec["cell_col"],
                "global_cell_row": rec["global_cell_row"],
                "global_cell_col": rec["global_cell_col"],
                "label": rec["label"],
                "slot_count": rec["slot_count"],
                "slot_index_1based": slot + 1,
                "embedding_row": emb_row,
                "fid_numeric": meta["fid_numeric"],
                "filename": meta["filename"],
                "relative_path": meta["relative_path"],
                "prediction_change_group": rec["prediction_change_group"],
                "prediction_change_subtype": rec["prediction_change_subtype"],
                "delta_gsv_logit_step16": rec["delta_gsv_logit"],
                "full_fusion_probability_reconstructed": float(full_recon_probs[i]),
                "full_fusion_prediction_frozen": full_pred,
                "loo_probability_without_image": p_without,
                "loo_prediction_without_image": pred_without,
                "loo_probability_shift": prob_shift,
                "loo_impact_logit": impact,
                "loo_abs_impact_logit": abs(impact),
                "prediction_flip_after_omission": int(pred_without != full_pred),
            }
            imagewise_rows.append(row)
            per_cell_image_rows[rec["covered_order"]].append(row)

    if len(imagewise_rows) != n_images:
        raise ValueError(
            f"Imagewise row count {len(imagewise_rows)} != {n_images}"
        )

    # Rank images within each cell by absolute omission impact.
    for rows in per_cell_image_rows.values():
        ranked = sorted(
            rows,
            key=lambda r: (
                -r["loo_abs_impact_logit"],
                r["slot_index_1based"],
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["abs_impact_rank_within_cell"] = rank

    # Preserve deterministic cell -> slot order in CSV.
    imagewise_rows.sort(
        key=lambda r: (
            r["covered_order"],
            r["slot_index_1based"],
        )
    )

    cellwise_rows = []
    for rec in records:
        rows = per_cell_image_rows[rec["covered_order"]]
        top = min(
            rows,
            key=lambda r: (
                -r["loo_abs_impact_logit"],
                r["slot_index_1based"],
            ),
        )
        impacts = np.asarray(
            [r["loo_impact_logit"] for r in rows],
            dtype=np.float64,
        )
        flip_count = int(
            sum(r["prediction_flip_after_omission"] for r in rows)
        )

        cellwise_rows.append(
            {
                "covered_order": rec["covered_order"],
                "patch_order": rec["patch_order"],
                "cell_row": rec["cell_row"],
                "cell_col": rec["cell_col"],
                "global_cell_row": rec["global_cell_row"],
                "global_cell_col": rec["global_cell_col"],
                "label": rec["label"],
                "slot_count": rec["slot_count"],
                "prediction_change_group": rec["prediction_change_group"],
                "prediction_change_subtype": rec["prediction_change_subtype"],
                "delta_gsv_logit_step16": rec["delta_gsv_logit"],
                "full_fusion_probability": float(full_recon_probs[i]),
                "loo_mean_impact_logit": float(np.mean(impacts)),
                "loo_median_impact_logit": float(np.median(impacts)),
                "loo_mean_abs_impact_logit": float(np.mean(np.abs(impacts))),
                "loo_max_abs_impact_logit": float(np.max(np.abs(impacts))),
                "loo_any_prediction_flip": int(flip_count > 0),
                "loo_prediction_flip_count": flip_count,
                "most_influential_slot_1based": top["slot_index_1based"],
                "most_influential_embedding_row": top["embedding_row"],
                "most_influential_filename": top["filename"],
                "most_influential_relative_path": top["relative_path"],
                "most_influential_signed_impact_logit": top["loo_impact_logit"],
                "most_influential_abs_impact_logit": top["loo_abs_impact_logit"],
            }
        )

    # Group summaries based on CELL-level max |LOO impact|.
    group_summary_rows = []
    subtypes = sorted(
        {r["prediction_change_subtype"] for r in cellwise_rows}
    )
    for subtype in subtypes:
        subset = [
            r for r in cellwise_rows
            if r["prediction_change_subtype"] == subtype
        ]
        max_abs = np.asarray(
            [r["loo_max_abs_impact_logit"] for r in subset],
            dtype=np.float64,
        )
        signed_top = np.asarray(
            [r["most_influential_signed_impact_logit"] for r in subset],
            dtype=np.float64,
        )
        group_summary_rows.append(
            {
                "prediction_change_subtype": subtype,
                "cells_k_ge_2": len(subset),
                "mean_max_abs_loo_impact": float(np.mean(max_abs)),
                "median_max_abs_loo_impact": float(np.median(max_abs)),
                "mean_signed_most_influential_impact": float(np.mean(signed_top)),
                "median_signed_most_influential_impact": float(np.median(signed_top)),
                "cells_with_any_prediction_flip": int(
                    sum(r["loo_any_prediction_flip"] for r in subset)
                ),
                "percent_cells_with_any_prediction_flip": float(
                    np.mean(
                        [r["loo_any_prediction_flip"] for r in subset]
                    ) * 100.0
                ),
            }
        )

    # K-wise summaries.
    k_summary_rows = []
    for kval in sorted({r["slot_count"] for r in cellwise_rows}):
        subset = [r for r in cellwise_rows if r["slot_count"] == kval]
        max_abs = np.asarray(
            [r["loo_max_abs_impact_logit"] for r in subset],
            dtype=np.float64,
        )
        k_summary_rows.append(
            {
                "K": kval,
                "cells": len(subset),
                "mean_max_abs_loo_impact": float(np.mean(max_abs)),
                "median_max_abs_loo_impact": float(np.median(max_abs)),
                "cells_with_any_prediction_flip": int(
                    sum(r["loo_any_prediction_flip"] for r in subset)
                ),
                "percent_cells_with_any_prediction_flip": float(
                    np.mean(
                        [r["loo_any_prediction_flip"] for r in subset]
                    ) * 100.0
                ),
            }
        )

    all_impacts = np.asarray(
        [r["loo_impact_logit"] for r in imagewise_rows],
        dtype=np.float64,
    )
    all_max_abs = np.asarray(
        [r["loo_max_abs_impact_logit"] for r in cellwise_rows],
        dtype=np.float64,
    )
    any_flip_cells = int(sum(r["loo_any_prediction_flip"] for r in cellwise_rows))

    # Commit only after all checks.
    write_csv(IMAGEWISE_CSV, imagewise_rows)
    write_csv(CELLWISE_CSV, cellwise_rows)
    write_csv(GROUP_SUMMARY_CSV, group_summary_rows)
    write_csv(K_SUMMARY_CSV, k_summary_rows)

    audit = {
        "status": "PASSED",
        "experiment": "XAI Step 17 leave-one-image-out GSV influence",
        "seed": SEED,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "model_modified": False,
        "population": {
            "covered_cells_total": N_COVERED,
            "k_eq_1_cells_excluded_from_loo": k1_count,
            "k_ge_2_cells_analyzed": n_cells,
            "real_images_analyzed": n_images,
        },
        "definition": (
            "LOO impact_i = logit(full Fusion) - logit(Fusion after omitting "
            "image i and recomputing masked mean over the remaining real images)"
        ),
        "interpretation": {
            "positive": (
                "Presence of image i pushes the fixed Fusion prediction toward informal "
                "relative to omitting that image."
            ),
            "negative": (
                "Presence of image i pushes the fixed Fusion prediction toward non-informal "
                "relative to omitting that image."
            ),
            "absolute": "Magnitude is prediction sensitivity/influence.",
            "not_claimed": (
                "LOO impact is not an additive attribution weight and not a causal effect."
            ),
        },
        "full_reconstruction": {
            "population": "full 5,744 covered cells before K filtering",
            "max_abs_probability_difference_vs_frozen_final_test": max_recon_diff,
            "mean_abs_probability_difference_vs_frozen_final_test": mean_recon_diff,
            "median_abs_probability_difference_vs_frozen_final_test": median_recon_diff,
            "q99_abs_probability_difference_vs_frozen_final_test": q99_recon_diff,
            "threshold_0p59_prediction_flip_count": recon_flip_count,
            "tolerance": FULL_RECON_TOL,
        },
        "image_level_impact": {
            "mean": float(np.mean(all_impacts)),
            "median": float(np.median(all_impacts)),
            "mean_absolute": float(np.mean(np.abs(all_impacts))),
            "median_absolute": float(np.median(np.abs(all_impacts))),
            "positive_percent": float(np.mean(all_impacts > 0) * 100.0),
            "negative_percent": float(np.mean(all_impacts < 0) * 100.0),
        },
        "cell_level": {
            "mean_max_abs_impact": float(np.mean(all_max_abs)),
            "median_max_abs_impact": float(np.median(all_max_abs)),
            "cells_with_any_prediction_flip_after_single_image_omission": any_flip_cells,
            "percent_cells_with_any_prediction_flip_after_single_image_omission": float(
                any_flip_cells / n_cells * 100.0
            ),
        },
        "outputs": {
            "imagewise_csv": str(IMAGEWISE_CSV),
            "cellwise_csv": str(CELLWISE_CSV),
            "group_summary_csv": str(GROUP_SUMMARY_CSV),
            "k_summary_csv": str(K_SUMMARY_CSV),
        },
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "=" * 96,
        "Nairobi v2 — XAI Step 17 LOO Audit",
        "=" * 96,
        "STATUS: PASSED",
        "Seed: 42",
        "Training performed: NO",
        "Threshold tuning performed: NO",
        "Model modification: NO",
        "",
        "Population:",
        f"  Covered cells total: {N_COVERED}",
        f"  K=1 cells excluded (within-cell LOO undefined): {k1_count}",
        f"  K>=2 cells analyzed: {n_cells}",
        f"  Real images analyzed: {n_images}",
        "",
        "Definition:",
        "  LOO impact_i = logit(full Fusion) - logit(Fusion without image i)",
        "  Masked mean is recomputed over the remaining K-1 real images.",
        "  Positive -> image pushes toward informal.",
        "  Negative -> image pushes toward non-informal.",
        "  NOT an additive weight and NOT a causal effect.",
        "",
        "Full-model reconstruction check:",
        "  Population/order: full 5,744 covered cells before K filtering",
        f"  Max probability difference vs frozen final test: {max_recon_diff:.10f}",
        f"  Mean probability difference vs frozen final test: {mean_recon_diff:.10f}",
        f"  Median probability difference vs frozen final test: {median_recon_diff:.10f}",
        f"  Q99 probability difference vs frozen final test: {q99_recon_diff:.10f}",
        f"  Threshold-0.59 prediction flips: {recon_flip_count}",
        "",
        "Image-level LOO impact:",
        f"  Mean impact: {float(np.mean(all_impacts)):.6f}",
        f"  Median impact: {float(np.median(all_impacts)):.6f}",
        f"  Mean |impact|: {float(np.mean(np.abs(all_impacts))):.6f}",
        f"  Median |impact|: {float(np.median(np.abs(all_impacts))):.6f}",
        f"  Impact > 0: {float(np.mean(all_impacts > 0) * 100.0):.2f}%",
        f"  Impact < 0: {float(np.mean(all_impacts < 0) * 100.0):.2f}%",
        "",
        "Cell-level sensitivity:",
        f"  Mean max |LOO impact| per cell: {float(np.mean(all_max_abs)):.6f}",
        f"  Median max |LOO impact| per cell: {float(np.median(all_max_abs)):.6f}",
        f"  Cells with >=1 prediction flip after omitting one image: {any_flip_cells}",
        f"  Percent of K>=2 cells with >=1 flip: {any_flip_cells / n_cells * 100.0:.2f}%",
        "",
        "Outputs:",
        f"  {IMAGEWISE_CSV}",
        f"  {CELLWISE_CSV}",
        f"  {GROUP_SUMMARY_CSV}",
        f"  {K_SUMMARY_CSV}",
    ]
    AUDIT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
