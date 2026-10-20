#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

PROJECT = Path.home() / "projects" / "nairobi_v2"
SEED = 42
OUT_DIR = PROJECT / "results/final_protocol_v2/seed_42"
OUT_JSON = OUT_DIR / "final_protocol_manifest_v2.json"
OUT_TXT = OUT_DIR / "final_protocol_manifest_v2.txt"

S2_MODEL = PROJECT / "models/s2_only_v2/seed_42/s2_only_model_best_v2.keras"
S2_ENCODER = PROJECT / "models/s2_only_v2/seed_42/s2_encoder_best_v2.keras"
S2_AUDIT = PROJECT / "results/s2_only_v2/seed_42/s2_only_training_audit_v2.json"
GSV_MODEL = PROJECT / "models/gsv_mean_pool_v2/seed_42/gsv_only_mean_pool_best_v2.keras"
GSV_ENCODER = PROJECT / "models/gsv_mean_pool_v2/seed_42/gsv_encoder_mean_pool_best_v2.keras"
GSV_AUDIT = PROJECT / "results/gsv_mean_pool_v2/seed_42/gsv_mean_pool_training_audit_v2.json"
FUSION_MODEL = PROJECT / "models/fusion_mean_pool_v2/seed_42/fusion_mean_pool_best_v2.keras"
FUSION_SCALER = PROJECT / "results/fusion_mean_pool_v2/seed_42/fusion_mean_pool_standardization_train_only_v2.npz"
FUSION_AUDIT = PROJECT / "results/fusion_mean_pool_v2/seed_42/fusion_mean_pool_training_audit_v2.json"
GSV_EMBED = PROJECT / "scratch/features/gsv_embeddings/resnet50_imagenet_v2/gsv_resnet50_embeddings_v2.npy"
GSV_EMBED_INDEX = PROJECT / "scratch/features/gsv_embeddings/resnet50_imagenet_v2/gsv_resnet50_embedding_index_v2.csv"
GSV_EMBED_AUDIT = PROJECT / "scratch/features/gsv_embeddings/resnet50_imagenet_v2/gsv_resnet50_embedding_audit_v2.json"
S2_NORM = PROJECT / "scratch/features/s2_inputs/s2_train_normalization_v2.json"
S2_INPUT_CONFIG = PROJECT / "scratch/features/s2_inputs/s2_input_rebuild_config_v2.json"

# IMPORTANT: these test artifacts are existence/size checked only. They are never opened.
TEST_ARTIFACTS = {
    "s2_input": PROJECT / "scratch/features/s2_inputs/X_test_s2_v2.npy",
    "labels": PROJECT / "metadata/latest_aligned_v2/test_label_patches_v2.npy",
    "valid_mask": PROJECT / "metadata/latest_aligned_v2/test_valid_label_mask_v2.npy",
    "full_gsv_manifest": PROJECT / "metadata/latest_full_gsv_aligned_v2/full_gsv_manifest_aligned_v2.csv",
}

EXPECTED_THRESHOLDS = {"s2_only": 0.78, "gsv_mean": 0.69, "fusion_mean": 0.59}
EXPECTED_PRETEST_COUNTS = {
    "test_patches": 102,
    "all_valid_test_cells": 22356,
    "gsv_covered_valid_test_cells": 5744,
    "no_gsv_valid_test_cells": 16612,
}
EXPECTED_GSV_EMBED_SHA256 = "cad7b6f4846ce957cecc004d9f5ec2f3a8c700eee47a4e127f2202cff1dd3e99"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stat_only(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    st = path.stat()
    return {"path": str(path), "exists": True, "size_bytes": int(st.st_size)}


def require_passed_training_audit(audit: dict, name: str) -> None:
    if str(audit.get("status", "")).upper() != "PASSED":
        raise ValueError(f"{name}: audit status is not PASSED")
    if int(audit.get("seed", -1)) != SEED:
        raise ValueError(f"{name}: audit seed is not {SEED}")
    if audit.get("test_data_loaded") is not False:
        raise ValueError(f"{name}: test_data_loaded is not False")
    if audit.get("test_data_evaluated") is not False:
        raise ValueError(f"{name}: test_data_evaluated is not False")


def extract_threshold(audit: dict, name: str) -> float:
    key = "selected_validation_threshold" if name == "s2_only" else "selected_threshold"
    obj = audit.get(key)
    if not isinstance(obj, dict) or "threshold" not in obj:
        raise ValueError(f"{name}: selected threshold missing from audit")
    value = float(obj["threshold"])
    expected = EXPECTED_THRESHOLDS[name]
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name}: threshold {value} != frozen expected {expected}")
    return value


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_JSON.exists() or OUT_TXT.exists():
        raise FileExistsError(
            "Final protocol is already frozen; refusing overwrite:\n"
            f"{OUT_JSON}\n{OUT_TXT}"
        )

    print("=" * 88)
    print("Nairobi v2 — Seed-42 Final Protocol Freeze / Pre-Test Integrity Audit")
    print("=" * 88)
    print("PRIMARY SEED: 42")
    print("THIS SCRIPT OPENS TEST DATA: NO")
    print("THIS SCRIPT EVALUATES TEST DATA: NO")

    s2_audit = load_json(S2_AUDIT)
    gsv_audit = load_json(GSV_AUDIT)
    fusion_audit = load_json(FUSION_AUDIT)
    require_passed_training_audit(s2_audit, "S2-only")
    require_passed_training_audit(gsv_audit, "GSV Mean")
    require_passed_training_audit(fusion_audit, "Fusion Mean")
    print("[1] Seed-42 training audits: PASSED")

    thresholds = {
        "s2_only": extract_threshold(s2_audit, "s2_only"),
        "gsv_mean": extract_threshold(gsv_audit, "gsv_mean"),
        "fusion_mean": extract_threshold(fusion_audit, "fusion_mean"),
    }
    print("[2] Frozen validation thresholds:", thresholds)

    frozen_files = {
        "s2_model": S2_MODEL,
        "s2_encoder": S2_ENCODER,
        "gsv_mean_model": GSV_MODEL,
        "gsv_mean_encoder": GSV_ENCODER,
        "fusion_mean_model": FUSION_MODEL,
        "fusion_train_only_scaler": FUSION_SCALER,
        "gsv_resnet50_embedding_index": GSV_EMBED_INDEX,
        "gsv_embedding_audit": GSV_EMBED_AUDIT,
        "s2_train_only_normalization": S2_NORM,
        "s2_input_rebuild_config": S2_INPUT_CONFIG,
    }
    missing = [p for p in frozen_files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen artifacts:\n" + "\n".join(str(p) for p in missing))
    hashes = {
        name: {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
        for name, path in frozen_files.items()
    }
    print("[3] Frozen model/preprocessing artifacts + checksums: PASSED")

    if not GSV_EMBED.is_file():
        raise FileNotFoundError(GSV_EMBED)
    embed_audit = load_json(GSV_EMBED_AUDIT)
    audit_sha = embed_audit.get("sha256") or embed_audit.get("embedding_sha256") or embed_audit.get("embeddings_sha256")
    if audit_sha is not None and str(audit_sha) != EXPECTED_GSV_EMBED_SHA256:
        raise ValueError("GSV embedding prior-audit SHA256 mismatch")
    gsv_embedding_record = {
        "path": str(GSV_EMBED),
        "size_bytes": int(GSV_EMBED.stat().st_size),
        "frozen_sha256_from_prior_audit": EXPECTED_GSV_EMBED_SHA256,
        "rehash_during_freeze": False,
    }
    print("[4] Frozen ResNet50 embedding artifact: PASSED")

    test_stats = {name: stat_only(path) for name, path in TEST_ARTIFACTS.items()}
    print("[5] Test artifact existence/size only: PASSED")
    print("    Test files opened: NO")
    print("    Test predictions/metrics computed: NO")

    protocol = {
        "status": "FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Nairobi v2",
        "primary_seed": 42,
        "primary_configuration_reason": (
            "Seed 42 is the pre-existing canonical experiment configuration; "
            "seeds 123 and 2026 are completed robustness diagnostics only and "
            "are not carried into final test or XAI."
        ),
        "final_models": {
            "s2_only": {
                "model_path": str(S2_MODEL), "encoder_path": str(S2_ENCODER),
                "threshold": thresholds["s2_only"], "prediction_population": "all valid cells"
            },
            "gsv_mean": {
                "model_path": str(GSV_MODEL), "encoder_path": str(GSV_ENCODER),
                "threshold": thresholds["gsv_mean"],
                "aggregation": "masked equal mean over real GSV slots only",
                "max_images_per_cell": 7, "padding_participates_in_mean": False,
                "prediction_population": "GSV-covered valid cells only"
            },
            "fusion_mean": {
                "model_path": str(FUSION_MODEL), "scaler_path": str(FUSION_SCALER),
                "threshold": thresholds["fusion_mean"],
                "architecture": (
                    "frozen S2 cell-256 + frozen GSV-Mean cell-256 -> separate train-only "
                    "standardization -> concat-512 -> Dense128 ReLU L2=1e-4 -> Dropout0.30 -> sigmoid"
                ),
                "prediction_population": "GSV-covered valid cells only"
            },
        },
        "missing_gsv_policy": {
            "missing_gsv_is_predictor": False,
            "zero_gsv_pseudo_input_used": False,
            "conditional_routing": {"gsv_available": "Fusion Mean", "gsv_unavailable": "S2-only"},
        },
        "final_test_plan": {
            "expected_pretest_counts": EXPECTED_PRETEST_COUNTS,
            "covered_subset_comparison": {
                "population": "5,744 GSV-covered valid test cells",
                "models": ["S2-only", "GSV Mean", "Fusion Mean"],
                "metrics": ["PR-AUC", "Informal precision", "Informal recall", "Informal F1", "Informal IoU", "Macro-F1"],
                "threshold_source": "validation only; frozen above",
            },
            "all_valid_comparison": {
                "population": "22,356 valid test cells",
                "systems": ["S2-only everywhere", "Conditional system: Fusion on covered cells, S2-only otherwise"],
                "conditional_system_metrics": ["Informal precision", "Informal recall", "Informal F1", "Informal IoU", "Macro-F1", "Accuracy"],
                "mixed_model_pr_auc": "NOT REPORTED",
            },
            "test_based_threshold_selection": False,
            "test_based_model_selection": False,
            "post_test_retraining_or_tuning": False,
        },
        "xai_target_after_test": {
            "model": "Fusion Mean seed 42",
            "core_steps": [
                "GSV modality neutralization / delta-logit",
                "S2-vs-Fusion corrected/degraded case analysis",
                "Leave-one-image-out omission impact for K>=2",
                "Fusion-targeted GSV Grad-CAM",
                "Semantic segmentation pilot subject to quality gate",
                "Spatial maps of contribution/correction/degradation",
            ],
        },
        "frozen_artifact_hashes": hashes,
        "gsv_embedding": gsv_embedding_record,
        "test_artifacts_stat_only": test_stats,
        "test_data_opened_by_this_script": False,
        "test_data_evaluated_by_this_script": False,
    }
    OUT_JSON.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    lines = [
        "=" * 88,
        "Nairobi v2 — Seed-42 Final Protocol Freeze",
        "=" * 88,
        "STATUS: FROZEN",
        "Primary seed: 42",
        "",
        f"S2-only threshold:  {thresholds['s2_only']}",
        f"GSV Mean threshold: {thresholds['gsv_mean']}",
        f"Fusion threshold:   {thresholds['fusion_mean']}",
        "",
        "Final GSV aggregation: MASKED MEAN POOLING",
        "Missing-GSV predictor: NO",
        "Zero-GSV pseudo-input: NO",
        "Routing: has GSV -> Fusion; no GSV -> S2-only",
        "",
        "Expected final-test counts from pre-test metadata audits:",
        f"  all valid: {EXPECTED_PRETEST_COUNTS['all_valid_test_cells']}",
        f"  GSV-covered: {EXPECTED_PRETEST_COUNTS['gsv_covered_valid_test_cells']}",
        f"  no-GSV: {EXPECTED_PRETEST_COUNTS['no_gsv_valid_test_cells']}",
        "",
        "Covered test: S2 vs GSV Mean vs Fusion Mean",
        "All-valid test: S2 everywhere vs conditional routed system",
        "Conditional mixed-model PR-AUC: NOT REPORTED",
        "",
        "Test-based threshold tuning: NO",
        "Test-based model selection: NO",
        "Post-test retraining/tuning: NO",
        "",
        "THIS FREEZE SCRIPT OPENED TEST DATA: NO",
        "THIS FREEZE SCRIPT EVALUATED TEST DATA: NO",
        "",
        "PRE-TEST PROTOCOL FREEZE: PASSED",
        f"Manifest: {OUT_JSON}",
    ]
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
