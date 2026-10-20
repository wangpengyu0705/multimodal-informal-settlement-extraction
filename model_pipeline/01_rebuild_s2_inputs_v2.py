#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import csv
import json
import math
import sys

import numpy as np
import rasterio


PROJECT = Path.home() / "projects" / "nairobi_v2"
S2 = PROJECT / "scratch/data/s2/Nrb_sentinel2_extent3857.tif"
META = PROJECT / "metadata/latest_aligned_v2"
PATCH_CSV = META / "patch_index_mapping.csv"
OUT = PROJECT / "scratch/features/s2_inputs"
OUT.mkdir(parents=True, exist_ok=True)

BANDS = [2, 3, 4, 8]
BAND_NAMES = ["B2", "B3", "B4", "B8"]
PATCH = 128

EXPECTED_PATCHES = {"train": 306, "val": 102, "test": 102}
EXPECTED_VALID = {"train": 67220, "val": 23024, "test": 22356}
EXPECTED_LABEL_COUNTS = {
    "train": {-1: 11116, 0: 62787, 1: 4433},
    "val": {-1: 3088, 0: 21566, 1: 1458},
    "test": {-1: 3756, 0: 20827, 1: 1529},
}


def need(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def read_mapping(cols: int):
    required = {
        "split",
        "patch_order",
        "original_patch_index",
        "patch_row",
        "patch_col",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }

    rows = []
    with PATCH_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing mapping columns: {sorted(missing)}")

        for row in reader:
            for key in [
                "patch_order",
                "original_patch_index",
                "patch_row",
                "patch_col",
            ]:
                row[key] = int(row[key])

            for key in ["xmin", "ymin", "xmax", "ymax"]:
                row[key] = float(row[key])

            rows.append(row)

    if len(rows) != 510:
        raise ValueError(f"Expected 510 selected patches, found {len(rows)}")

    if len({(r["patch_row"], r["patch_col"]) for r in rows}) != 510:
        raise ValueError("Duplicate patch locations")

    split_rows = {}

    for split, expected_n in EXPECTED_PATCHES.items():
        subset = sorted(
            [r for r in rows if r["split"] == split],
            key=lambda r: r["patch_order"],
        )

        if len(subset) != expected_n:
            raise ValueError(
                f"{split}: expected {expected_n}, found {len(subset)}"
            )

        if [r["patch_order"] for r in subset] != list(range(expected_n)):
            raise ValueError(f"{split}: non-contiguous patch_order")

        for r in subset:
            expected_original_index = r["patch_row"] * cols + r["patch_col"]
            if r["original_patch_index"] != expected_original_index:
                raise ValueError(
                    f"{split} order {r['patch_order']}: "
                    "original_patch_index mismatch"
                )

        split_rows[split] = subset

    return rows, split_rows


def validate_labels(split: str):
    label_file = META / f"{split}_label_patches_v2.npy"
    mask_file = META / f"{split}_valid_label_mask_v2.npy"

    need(label_file)
    need(mask_file)

    labels = np.load(label_file, allow_pickle=False)
    mask = np.load(mask_file, allow_pickle=False).astype(bool)

    expected_shape = (EXPECTED_PATCHES[split], 16, 16)

    if labels.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError(
            f"{split}: label/mask shape mismatch: "
            f"{labels.shape}, {mask.shape}, expected {expected_shape}"
        )

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({-1, 0, 1}):
        raise ValueError(f"{split}: unexpected labels {sorted(unique_labels)}")

    recomputed_mask = np.isin(labels, [0, 1])

    if not np.array_equal(mask, recomputed_mask):
        raise ValueError(f"{split}: valid mask does not match labels")

    valid_count = int(recomputed_mask.sum())
    counts = {
        value: int((labels == value).sum())
        for value in [-1, 0, 1]
    }

    if valid_count != EXPECTED_VALID[split]:
        raise ValueError(
            f"{split}: valid count {valid_count} != {EXPECTED_VALID[split]}"
        )

    if counts != EXPECTED_LABEL_COUNTS[split]:
        raise ValueError(
            f"{split}: label counts {counts} != {EXPECTED_LABEL_COUNTS[split]}"
        )

    return {
        "shape": list(labels.shape),
        "valid": valid_count,
        "counts": {str(k): v for k, v in counts.items()},
    }


def build_train_spatial_mask(
    train_rows,
    raster_height: int,
    raster_width: int,
) -> np.ndarray:
    """
    True only for real raster pixels inside the frozen 306 TRAIN patches.
    Artificial bottom/right padding is excluded.
    """
    train_mask = np.zeros(
        (raster_height, raster_width),
        dtype=bool,
    )

    for row in train_rows:
        r0 = row["patch_row"] * PATCH
        c0 = row["patch_col"] * PATCH
        r1 = min(r0 + PATCH, raster_height)
        c1 = min(c0 + PATCH, raster_width)

        if r0 >= raster_height or c0 >= raster_width:
            continue

        train_mask[r0:r1, c0:c1] = True

    return train_mask


need(S2)
need(PATCH_CSV)

print("=" * 80)
print("Nairobi v2 — rebuild S2 inputs")
print("=" * 80)
print("S2:", S2)
print("Mapping:", PATCH_CSV)
print("Output:", OUT)


with rasterio.open(S2) as src:
    h = src.height
    w = src.width
    count = src.count
    crs = src.crs
    tr = src.transform
    bounds = src.bounds
    nodata = src.nodata

print("\n[1] Raster:", (h, w), "bands=", count, "CRS=", crs)
print("transform=", tr)
print("bounds=", bounds)
print("nodata=", nodata)

if (h, w, count) != (3160, 4897, 16):
    raise ValueError(f"Unexpected raster geometry/count: {(h, w, count)}")

if str(crs) != "EPSG:3857":
    raise ValueError(f"Unexpected CRS: {crs}")

for actual, expected, name in [
    (tr.a, 10.0, "pixel x"),
    (tr.e, -10.0, "pixel y"),
    (tr.c, 4081450.0, "origin x"),
    (tr.f, -129210.0, "origin y"),
]:
    if not np.isclose(actual, expected, atol=1e-8, rtol=0):
        raise ValueError(f"Unexpected {name}: {actual}")

rows = math.ceil(h / PATCH)
cols = math.ceil(w / PATCH)
ph = rows * PATCH
pw = cols * PATCH

if (rows, cols, ph, pw) != (25, 39, 3200, 4992):
    raise ValueError(
        f"Unexpected patch grid/padded shape: {(rows, cols, ph, pw)}"
    )

print(
    "patch grid=", (rows, cols),
    "full patches=", rows * cols,
    "padded=", (ph, pw),
)
print("Raster validation: PASSED")


all_rows, split_rows = read_mapping(cols)

for row in all_rows:
    expected_xmin = tr.c + row["patch_col"] * PATCH * tr.a
    expected_xmax = expected_xmin + PATCH * tr.a
    expected_ymax = tr.f + row["patch_row"] * PATCH * tr.e
    expected_ymin = expected_ymax - PATCH * abs(tr.e)

    for key, actual, expected in [
        ("xmin", row["xmin"], expected_xmin),
        ("xmax", row["xmax"], expected_xmax),
        ("ymin", row["ymin"], expected_ymin),
        ("ymax", row["ymax"], expected_ymax),
    ]:
        if not np.isclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError(
                f"{row['split']} order {row['patch_order']}: {key} mismatch"
            )

print(
    "\n[2] Frozen split:",
    {s: len(split_rows[s]) for s in split_rows},
)
print("No train_test_split is executed. PASSED")


label_audit = {
    split: validate_labels(split)
    for split in ["train", "val", "test"]
}

print("\n[3] Existing labels/masks validated (not regenerated):")
for split, audit in label_audit.items():
    print(split, audit)


print("\n[4] Build TRAIN-only normalization fit mask")

train_spatial_mask = build_train_spatial_mask(
    split_rows["train"],
    h,
    w,
)

train_real_pixel_count = int(train_spatial_mask.sum())

if train_real_pixel_count <= 0:
    raise ValueError("TRAIN spatial mask contains no real raster pixels")

print("Real raster pixels inside frozen TRAIN patches:", train_real_pixel_count)
print("Artificial bottom/right padding excluded from fitting: YES")
print("TRAIN spatial mask: PASSED")


print("\n[5] Train-only Sentinel-2 normalization")
print("Normalization fit source: TRAIN SPLIT ONLY")
print(
    "Validation/test use the SAME train-derived P2/P98 parameters: YES"
)

padded = np.zeros(
    (ph, pw, len(BANDS)),
    dtype=np.float32,
)

norm_stats = {}

with rasterio.open(S2) as src:
    for channel, (band_index, band_name) in enumerate(
        zip(BANDS, BAND_NAMES)
    ):
        band_ma = src.read(
            band_index,
            masked=True,
        ).astype(np.float32)

        band_data = np.asarray(
            band_ma.data,
            dtype=np.float32,
        )

        invalid_mask = np.ma.getmaskarray(band_ma)

        train_valid_mask = (
            train_spatial_mask
            & (~invalid_mask)
            & np.isfinite(band_data)
        )

        train_values = band_data[train_valid_mask]

        if train_values.size == 0:
            raise ValueError(
                f"{band_name}: no valid TRAIN pixels available"
            )

        p2 = float(np.percentile(train_values, 2))
        p98 = float(np.percentile(train_values, 98))

        if not np.isfinite([p2, p98]).all() or p98 <= p2:
            raise ValueError(
                f"{band_name}: invalid TRAIN percentile range"
            )

        full_valid_mask = (
            (~invalid_mask)
            & np.isfinite(band_data)
        )

        normalized_band = np.zeros(
            (h, w),
            dtype=np.float32,
        )

        normalized_values = (
            band_data[full_valid_mask] - p2
        ) / (p98 - p2)

        normalized_values = np.clip(
            normalized_values,
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        normalized_band[full_valid_mask] = normalized_values

        padded[:h, :w, channel] = normalized_band

        norm_stats[band_name] = {
            "source_band": band_index,
            "fit_split": "train",
            "train_valid_pixel_count": int(train_values.size),
            "p2": p2,
            "p98": p98,
            "full_raster_valid_pixel_count": int(full_valid_mask.sum()),
            "raster_masked_pixel_count": int(invalid_mask.sum()),
            "normalized_valid_min": float(normalized_values.min()),
            "normalized_valid_max": float(normalized_values.max()),
            "normalized_valid_mean": float(normalized_values.mean()),
            "normalized_valid_std": float(normalized_values.std()),
        }

        print(
            f"{band_name}: "
            f"train_valid={train_values.size:,}, "
            f"p2={p2:.6f}, p98={p98:.6f}, "
            f"min={normalized_values.min():.3f}, "
            f"max={normalized_values.max():.3f}"
        )

        del band_ma
        del band_data
        del invalid_mask
        del train_valid_mask
        del train_values
        del full_valid_mask
        del normalized_band
        del normalized_values

print("Train-only normalization: PASSED")


normalization_file = OUT / "s2_train_normalization_v2.json"

normalization_payload = {
    "created_at": datetime.now().isoformat(),
    "method": "per-band P2/P98 percentile normalization",
    "fit_split": "train",
    "bands": BANDS,
    "band_names": BAND_NAMES,
    "lower_percentile": 2,
    "upper_percentile": 98,
    "clip_range": [0.0, 1.0],
    "raster_nodata": (
        None if nodata is None else float(nodata)
    ),
    "raster_mask_used": True,
    "artificial_padding_used_for_fit": False,
    "statistics": norm_stats,
}

normalization_file.write_text(
    json.dumps(normalization_payload, indent=2),
    encoding="utf-8",
)

print("Saved train-only normalization:", normalization_file)


print("\n[6] Extract exact 510 patches in frozen patch_order")

index_rows = []
saved = {}

for split in ["train", "val", "test"]:
    subset = split_rows[split]

    X = np.empty(
        (len(subset), PATCH, PATCH, len(BANDS)),
        dtype=np.float32,
    )

    for row in subset:
        r0 = row["patch_row"] * PATCH
        c0 = row["patch_col"] * PATCH

        patch = padded[
            r0:r0 + PATCH,
            c0:c0 + PATCH,
            :,
        ]

        if patch.shape != (128, 128, 4):
            raise ValueError(
                f"{split} order {row['patch_order']}: "
                f"patch shape {patch.shape}"
            )

        X[row["patch_order"]] = patch

        index_rows.append(
            {
                key: row[key]
                for key in [
                    "split",
                    "patch_order",
                    "original_patch_index",
                    "patch_row",
                    "patch_col",
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                ]
            }
        )

    expected_shape = (
        EXPECTED_PATCHES[split],
        128,
        128,
        4,
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"{split}: X shape {X.shape} != {expected_shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"{split}: non-finite values in X")

    if float(X.min()) < 0.0 or float(X.max()) > 1.0:
        raise ValueError(
            f"{split}: normalized values outside [0,1]"
        )

    output_file = OUT / f"X_{split}_s2_v2.npy"

    np.save(
        output_file,
        X,
        allow_pickle=False,
    )

    reloaded = np.load(
        output_file,
        mmap_mode="r",
        allow_pickle=False,
    )

    if (
        reloaded.shape != expected_shape
        or reloaded.dtype != np.float32
    ):
        raise ValueError(
            f"{split}: saved file reload failed"
        )

    saved[split] = {
        "shape": list(reloaded.shape),
        "dtype": str(reloaded.dtype),
        "bytes": int(output_file.stat().st_size),
        "min": float(X.min()),
        "max": float(X.max()),
    }

    print(
        split,
        X.shape,
        X.dtype,
        f"min={X.min():.3f}",
        f"max={X.max():.3f}",
        "saved=", output_file,
    )

    del X
    del reloaded

print("Extraction + saved-file reload: PASSED")


index_file = OUT / "s2_input_index_v2.csv"

index_fields = [
    "split",
    "patch_order",
    "original_patch_index",
    "patch_row",
    "patch_col",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
]

with index_file.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=index_fields,
    )
    writer.writeheader()
    writer.writerows(index_rows)


config = {
    "created_at": datetime.now().isoformat(),
    "python": sys.executable,
    "s2": str(S2),
    "mapping": str(PATCH_CSV),
    "bands": BANDS,
    "band_names": BAND_NAMES,
    "normalization": {
        "method": "P2/P98 then clip [0,1]",
        "fit_split": "train",
        "validation_test_use_train_parameters": True,
        "raster_mask_used": True,
        "artificial_padding_used_for_fit": False,
        "statistics": norm_stats,
    },
    "raster_shape": [h, w],
    "patch_grid": [rows, cols],
    "padded_shape": [ph, pw],
    "selected_patch_counts": EXPECTED_PATCHES,
    "random_split_created": False,
    "labels_regenerated": False,
    "label_audit": label_audit,
    "saved_arrays": saved,
}

config_file = OUT / "s2_input_rebuild_config_v2.json"

config_file.write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)


audit = [
    "=" * 80,
    "Nairobi v2 — S2 Input Rebuild Audit",
    "=" * 80,
    f"S2: {S2}",
    f"Mapping: {PATCH_CSV}",
    f"Raster: {(h, w)}, bands={count}, CRS={crs}",
    f"Patch grid: {(rows, cols)}; padded={(ph, pw)}",
    "Bands: B2/B3/B4/B8",
    "Normalization fit source: TRAIN SPLIT ONLY",
    "Normalization transform: same TRAIN-derived P2/P98 applied to train/val/test",
    "Raster masked/NoData pixels excluded from percentile fitting: YES",
    "Artificial bottom/right padding excluded from percentile fitting: YES",
    "Numeric value 0 is not treated as NoData unless masked by raster: YES",
    "Split: train=306, val=102, test=102; NO new random split",
    "Existing aligned-v2 labels/masks validated; NOT regenerated",
    "",
]

for band_name in BAND_NAMES:
    stats = norm_stats[band_name]
    audit.append(
        f"{band_name}: "
        f"train_valid_pixels={stats['train_valid_pixel_count']}; "
        f"P2={stats['p2']}; "
        f"P98={stats['p98']}"
    )

audit.append("")

for split in ["train", "val", "test"]:
    audit.append(
        f"{split}: "
        f"X={saved[split]['shape']}; "
        f"valid labels={label_audit[split]['valid']}; "
        f"label counts={label_audit[split]['counts']}"
    )

audit += [
    "",
    "Raster validation: PASSED",
    "Frozen split/order validation: PASSED",
    "Label/mask validation: PASSED",
    "Train-only normalization: PASSED",
    "Extraction/reload validation: PASSED",
    "",
    "S2 INPUT REBUILD: PASSED",
]

audit_text = "\n".join(audit)

audit_file = OUT / "s2_input_rebuild_audit_v2.txt"

audit_file.write_text(
    audit_text,
    encoding="utf-8",
)

print("\n" + audit_text)
print("\nSaved:", index_file)
print("Saved:", normalization_file)
print("Saved:", config_file)
print("Saved:", audit_file)
