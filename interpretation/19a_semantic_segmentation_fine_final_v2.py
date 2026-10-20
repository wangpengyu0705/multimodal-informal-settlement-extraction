#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


PROJECT = Path.home() / "projects" / "nairobi_v2"
SEED = 42

MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
LOCAL_MODEL_DIR = PROJECT / "models/pretrained/segformer_b2_ade20k_v2"

STEP18_HIGH = (
    PROJECT / "results/xai_gradcam_v2/seed_42/representative_high_impact_v2"
    / "representative_high_impact_metrics_v2.csv"
)
STEP18_MEDIAN = (
    PROJECT / "results/xai_gradcam_v2/seed_42/representative_median_v2"
    / "representative_median_metrics_v2.csv"
)

OUT_DIR = (
    PROJECT / "results/xai_semantic_v2/seed_42"
    / "19a_fine_semantic_segmentation"
)
MASK_DIR = OUT_DIR / "masks_fine_ade20k"
COLOR_DIR = OUT_DIR / "segmentation_fine_color"
OVERLAY_DIR = OUT_DIR / "segmentation_fine_overlay"

CASE_CSV = OUT_DIR / "fine_semantic_case_summary_final_v2.csv"
CLASS_CSV = OUT_DIR / "fine_semantic_class_fractions_final_v2.csv"
PRESENT_CSV = OUT_DIR / "fine_semantic_classes_present_final_v2.csv"
AUDIT_JSON = OUT_DIR / "fine_semantic_audit_final_v2.json"
NOTES_TXT = OUT_DIR / "fine_semantic_notes_final_v2.txt"

QC_MEDIAN = OUT_DIR / "qc_fine_semantic_median_final_v2.png"
QC_HIGH = OUT_DIR / "qc_fine_semantic_high_impact_final_v2.png"

TILE_SIZE = 128
TILE_STRIDE = 96
OVERLAY_ALPHA = 0.48

SUBTYPE_ORDER = [
    "fp_to_correct",
    "fn_to_correct",
    "correct_to_fp",
    "correct_to_fn",
]
SUBTYPE_LABEL = {
    "fp_to_correct": "S2 FP → Fusion correct",
    "fn_to_correct": "S2 FN → Fusion correct",
    "correct_to_fp": "S2 correct → Fusion FP",
    "correct_to_fn": "S2 correct → Fusion FN",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    return p.parse_args()


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_existing_path(value: str) -> Path:
    p = Path(str(value))
    if p.is_file():
        return p
    marker = "projects/nairobi_v2/"
    s = str(p)
    if marker in s:
        q = PROJECT / s.split(marker, 1)[1]
        if q.is_file():
            return q
    return p


def canonical_subtype(x: str) -> str:
    s = str(x).strip().lower()
    mapping = {
        "fp_to_correct": "fp_to_correct",
        "s2_fp_to_correct": "fp_to_correct",
        "fn_to_correct": "fn_to_correct",
        "s2_fn_to_correct": "fn_to_correct",
        "correct_to_fp": "correct_to_fp",
        "s2_correct_to_fusion_fp": "correct_to_fp",
        "correct_to_fn": "correct_to_fn",
        "s2_correct_to_fusion_fn": "correct_to_fn",
    }
    if s not in mapping:
        raise ValueError(f"Unknown subtype: {x}")
    return mapping[s]


def canonical_k(x: str) -> str:
    s = str(x).replace(" ", "")
    if s in {"K1", "K=1", "1"}:
        return "K1"
    if s in {"K>=2", "K≥2", ">=2", "≥2"}:
        return "K>=2"
    raise ValueError(f"Unknown K stratum: {x}")


def load_case_records():
    records = []
    for selection, path in [
        ("high_impact", STEP18_HIGH),
        ("median", STEP18_MEDIAN),
    ]:
        rows = read_csv(path)
        if len(rows) != 8:
            raise ValueError(f"{selection}: expected 8 frozen Step-18 cases, found {len(rows)}")

        for r in rows:
            model_input_png = resolve_existing_path(r["original_png"])
            heatmap_npy = resolve_existing_path(r["heatmap_npy"])
            raw_image_path = resolve_existing_path(r["image_path"]) if r.get("image_path") else None

            if not model_input_png.is_file():
                raise FileNotFoundError(model_input_png)
            if not heatmap_npy.is_file():
                raise FileNotFoundError(heatmap_npy)

            records.append(
                {
                    "selection": selection,
                    "subtype": canonical_subtype(r["subtype"]),
                    "k_stratum": canonical_k(r["k_stratum"]),
                    "covered_order": int(float(r["covered_order"])),
                    "target_embedding_row": int(float(r["target_embedding_row"])),
                    "model_input_png": model_input_png,
                    "heatmap_npy": heatmap_npy,
                    "raw_image_path": raw_image_path if raw_image_path and raw_image_path.is_file() else None,
                }
            )

    if len(records) != 16:
        raise ValueError(f"Expected 16 total frozen cases, found {len(records)}")
    return records


def tile_starts(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def color_for_class(class_id: int):
    # Deterministic QC palette only; not claimed as official ADE20K colors.
    import colorsys
    golden = 0.618033988749895
    h = (class_id * golden) % 1.0
    s = 0.62 + 0.20 * ((class_id % 3) / 2.0)
    v = 0.84 + 0.10 * (class_id % 2)
    r, g, b = colorsys.hsv_to_rgb(h, min(s, 0.88), min(v, 0.96))
    return tuple(int(round(255 * z)) for z in (r, g, b))


@torch.inference_mode()
def segment_model_input(image: Image.Image, processor, model):
    image = image.convert("RGB")
    arr = np.asarray(image)
    h, w = arr.shape[:2]

    if (h, w) != (128, 500):
        raise ValueError(
            f"Expected exact Step-18 model-input grid (128, 500), got {(h, w)}. "
            "This final workflow does not silently resize the panorama."
        )

    starts = tile_starts(w, TILE_SIZE, TILE_STRIDE)
    num_labels = int(model.config.num_labels)

    logits_sum = np.zeros((num_labels, h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    x = np.linspace(-1.0, 1.0, TILE_SIZE, dtype=np.float32)
    wx = 0.25 + 0.75 * (0.5 + 0.5 * np.cos(np.pi * x))
    weight_2d = np.tile(wx[None, :], (TILE_SIZE, 1)).astype(np.float32)

    for start in starts:
        crop = Image.fromarray(arr[:, start:start + TILE_SIZE, :])
        inputs = processor(images=crop, return_tensors="pt")
        outputs = model(**inputs)
        logits = F.interpolate(
            outputs.logits,
            size=(TILE_SIZE, TILE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0].cpu().numpy().astype(np.float32)

        end = start + TILE_SIZE
        logits_sum[:, :, start:end] += logits * weight_2d[None, :, :]
        weight_sum[:, start:end] += weight_2d

    logits_avg = logits_sum / np.maximum(weight_sum[None, :, :], 1e-8)
    probs = torch.softmax(torch.from_numpy(logits_avg), dim=0).numpy()

    pred = np.argmax(probs, axis=0).astype(np.uint8)
    conf = np.max(probs, axis=0).astype(np.float32)
    return pred, conf, starts


def render_segmentation(mask: np.ndarray):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid in np.unique(mask):
        out[mask == cid] = color_for_class(int(cid))
    return out


def make_overlay(rgb: np.ndarray, seg_rgb: np.ndarray):
    out = (
        (1.0 - OVERLAY_ALPHA) * rgb.astype(np.float32)
        + OVERLAY_ALPHA * seg_rgb.astype(np.float32)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def top_classes(mask, id2label, n=4):
    ids, counts = np.unique(mask, return_counts=True)
    total = float(mask.size)
    items = [
        (int(cid), id2label.get(int(cid), str(int(cid))), int(cnt) / total)
        for cid, cnt in zip(ids, counts)
    ]
    return sorted(items, key=lambda x: x[2], reverse=True)[:n]


def case_stem(r):
    k = "k1" if r["k_stratum"] == "K1" else "kge2"
    return (
        f'{r["selection"]}__{r["subtype"]}__{k}'
        f'__order{r["covered_order"]}__emb{r["target_embedding_row"]}'
    )


def build_qc_panel(selection, records, summaries):
    rec = {
        (r["subtype"], r["k_stratum"]): r
        for r in records if r["selection"] == selection
    }
    summ = {
        (r["subtype"], r["k_stratum"]): r
        for r in summaries if r["selection"] == selection
    }

    # Step-18-style layout: fixed publication width, independent label column,
    # no text placed underneath image axes, generous vertical spacing.
    fig = plt.figure(figsize=(190 / 25.4, 112 / 25.4), dpi=180)
    gs = fig.add_gridspec(
        6, 5,
        width_ratios=[0.78, 1, 1, 1, 1],
        height_ratios=[0.18, 0.20, 1, 1, 1, 1],
        left=0.035, right=0.995, top=0.975, bottom=0.045,
        wspace=0.065, hspace=0.22,
    )

    ax = fig.add_subplot(gs[0, 1:3]); ax.axis("off")
    ax.text(0.5, 0.45, "K = 1", ha="center", va="center", fontsize=9, fontweight="bold")
    ax = fig.add_subplot(gs[0, 3:5]); ax.axis("off")
    ax.text(0.5, 0.45, "K ≥ 2", ha="center", va="center", fontsize=9, fontweight="bold")

    headers = ["Original", "Semantic overlay", "Original", "Semantic overlay"]
    for j, txt in enumerate(headers, start=1):
        ax = fig.add_subplot(gs[1, j]); ax.axis("off")
        ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=7.5)

    for i, subtype in enumerate(SUBTYPE_ORDER):
        row = i + 2
        axl = fig.add_subplot(gs[row, 0]); axl.axis("off")
        axl.text(
            0.98, 0.5,
            f"({chr(97+i)}) {SUBTYPE_LABEL[subtype]}",
            ha="right", va="center", fontsize=7.0,
            fontweight="bold", wrap=True,
        )

        for k_idx, kstr in enumerate(["K1", "K>=2"]):
            rr = rec[(subtype, kstr)]
            ss = summ[(subtype, kstr)]
            original = np.asarray(Image.open(rr["model_input_png"]).convert("RGB"))
            overlay = np.asarray(Image.open(ss["overlay_png"]).convert("RGB"))

            c0 = 1 + 2 * k_idx
            for col, img in [(c0, original), (c0 + 1, overlay)]:
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(img)
                ax.set_axis_off()

    out = QC_MEDIAN if selection == "median" else QC_HIGH
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    if not args.run:
        raise SystemExit("Run with --run.")

    if not LOCAL_MODEL_DIR.is_dir():
        raise RuntimeError(
            f"Cached SegFormer-B2 model not found: {LOCAL_MODEL_DIR}\n"
            "Keep the existing model cache; Step 19 cleanup should not delete it."
        )

    print("=" * 96)
    print("Nairobi v2 — Step 19a Final Fine Semantic Segmentation")
    print("=" * 96)
    print("Method: direct segmentation on the exact 500×128 Step-18 model-input RGB")
    print("Cases: same 16 frozen representative Grad-CAM target images")
    print("New training/fine-tuning: NO")
    print("Model/threshold tuning: NO")
    print("Grad-CAM recomputation: NO")
    print(f"Model: {MODEL_ID}")
    print(f"Tiling: {TILE_SIZE}×{TILE_SIZE}, stride {TILE_STRIDE}, weighted-logit stitching")
    print()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for d in [MASK_DIR, COLOR_DIR, OVERLAY_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    processor = AutoImageProcessor.from_pretrained(
        LOCAL_MODEL_DIR, local_files_only=True, use_fast=False
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        LOCAL_MODEL_DIR, local_files_only=True
    )
    model.eval().to("cpu")
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    records = load_case_records()
    summaries, class_rows = [], []
    total_counts = Counter()
    total_pixels = 0

    for idx, r in enumerate(records, start=1):
        print(
            f'[{idx:02d}/16] {r["selection"]} | {r["subtype"]} | '
            f'{r["k_stratum"]} | order={r["covered_order"]}'
        )

        image = Image.open(r["model_input_png"]).convert("RGB")
        original = np.asarray(image)

        heat = np.squeeze(np.asarray(np.load(r["heatmap_npy"])))
        if heat.ndim != 2:
            raise ValueError(f"Grad-CAM must be 2-D after squeeze: {r['heatmap_npy']} {heat.shape}")
        expected = (image.height, image.width)
        if tuple(heat.shape) != expected:
            raise ValueError(
                "Model-input / Grad-CAM alignment failure.\n"
                f"  RGB: {r['model_input_png']} -> {expected}\n"
                f"  Grad-CAM: {r['heatmap_npy']} -> {tuple(heat.shape)}"
            )

        pred, conf, starts = segment_model_input(image, processor, model)
        if pred.shape != expected:
            raise ValueError(f"Semantic mask shape {pred.shape} != expected {expected}")

        seg_rgb = render_segmentation(pred)
        overlay = make_overlay(original, seg_rgb)

        stem = case_stem(r)
        mask_path = MASK_DIR / f"{stem}__mask.npy"
        color_path = COLOR_DIR / f"{stem}__segmentation.png"
        overlay_path = OVERLAY_DIR / f"{stem}__overlay.png"

        np.save(mask_path, pred)
        Image.fromarray(seg_rgb).save(color_path)
        Image.fromarray(overlay).save(overlay_path)

        ids, counts = np.unique(pred, return_counts=True)
        total = int(pred.size)
        total_pixels += total
        for cid, cnt in zip(ids, counts):
            cid, cnt = int(cid), int(cnt)
            total_counts[cid] += cnt
            class_rows.append(
                {
                    "selection": r["selection"],
                    "subtype": r["subtype"],
                    "k_stratum": r["k_stratum"],
                    "covered_order": r["covered_order"],
                    "target_embedding_row": r["target_embedding_row"],
                    "class_id": cid,
                    "class_label": id2label.get(cid, str(cid)),
                    "pixel_count": cnt,
                    "pixel_fraction": cnt / float(total),
                }
            )

        tops = top_classes(pred, id2label, n=4)
        summaries.append(
            {
                "selection": r["selection"],
                "subtype": r["subtype"],
                "k_stratum": r["k_stratum"],
                "covered_order": r["covered_order"],
                "target_embedding_row": r["target_embedding_row"],
                "model_input_png": str(r["model_input_png"]),
                "raw_image_path": str(r["raw_image_path"]) if r["raw_image_path"] else "",
                "gradcam_heatmap_npy": str(r["heatmap_npy"]),
                "native_width": int(image.width),
                "native_height": int(image.height),
                "gradcam_width": int(heat.shape[1]),
                "gradcam_height": int(heat.shape[0]),
                "pixel_grid_alignment": "PASSED",
                "tile_starts": ",".join(map(str, starts)),
                "mean_max_softmax": float(np.mean(conf)),
                "median_max_softmax": float(np.median(conf)),
                "fraction_conf_ge_0_50": float(np.mean(conf >= 0.50)),
                "fraction_conf_ge_0_70": float(np.mean(conf >= 0.70)),
                "dominant_class": tops[0][1] if tops else "",
                "dominant_fraction": tops[0][2] if tops else 0.0,
                "top3_display": " | ".join(
                    f"{label} {100*frac:.0f}%" for _, label, frac in tops[:3]
                ),
                "mask_npy": str(mask_path),
                "segmentation_png": str(color_path),
                "overlay_png": str(overlay_path),
            }
        )

    write_csv(CASE_CSV, summaries)
    write_csv(CLASS_CSV, class_rows)

    present_rows = []
    for cid, cnt in sorted(total_counts.items(), key=lambda kv: kv[1], reverse=True):
        present_rows.append(
            {
                "class_id": cid,
                "class_label": id2label.get(cid, str(cid)),
                "total_pixel_count": cnt,
                "total_pixel_fraction": cnt / float(total_pixels),
            }
        )
    write_csv(PRESENT_CSV, present_rows)

    build_qc_panel("median", records, summaries)
    build_qc_panel("high_impact", records, summaries)

    audit = {
        "status": "PASSED",
        "workflow": "Step19a_final_fine_semantic_segmentation",
        "model_id": MODEL_ID,
        "n_cases": len(summaries),
        "semantic_input": "exact Step-18 model-input RGB",
        "expected_grid": [128, 500],
        "all_pixel_grid_alignment_passed": all(
            s["pixel_grid_alignment"] == "PASSED" for s in summaries
        ),
        "training_or_finetuning": False,
        "model_or_threshold_tuning": False,
        "gradcam_recomputed": False,
        "mean_case_mean_max_softmax": float(np.mean([s["mean_max_softmax"] for s in summaries])),
        "mean_fraction_conf_ge_0_50": float(np.mean([s["fraction_conf_ge_0_50"] for s in summaries])),
        "mean_fraction_conf_ge_0_70": float(np.mean([s["fraction_conf_ge_0_70"] for s in summaries])),
        "method_note": (
            "Final Step-19 workflow uses direct SegFormer-B2 segmentation on the exact "
            "500×128 Step-18 model-input grid."
        ),
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    NOTES_TXT.write_text(
        "\n".join([
            "Nairobi v2 — Step 19a Final Fine Semantic Segmentation",
            "STATUS: PASSED",
            "",
            "Selected method:",
            "  Direct SegFormer-B2 segmentation on exact Step-18 500×128 model-input RGB.",
            "  128×128 overlapping tiles, stride 96, weighted class-logit stitching.",
            "",
            "Scope:",
            "  Same 16 frozen representative Step-18 Grad-CAM target images.",
            "  No training/fine-tuning/model selection/threshold tuning/Grad-CAM recomputation.",
            "",
            f"Outputs: {OUT_DIR}",
        ]) + "\n",
        encoding="utf-8",
    )

    print()
    print("STATUS: PASSED")
    print(f"Output: {OUT_DIR}")
    print(f"Mean(case mean max-softmax): {audit['mean_case_mean_max_softmax']:.4f}")
    print(f"Mean fraction pixels >= 0.50: {audit['mean_fraction_conf_ge_0_50']:.4f}")
    print(f"Mean fraction pixels >= 0.70: {audit['mean_fraction_conf_ge_0_70']:.4f}")


if __name__ == "__main__":
    main()
