#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform

PROJECT = Path.home() / "projects" / "nairobi_v2"
MAPPING_CSV = PROJECT / "metadata/latest_aligned_v2/cell_index_mapping_v2.csv"
CONFIG_JSON = PROJECT / "metadata/latest_aligned_v2/aligned_mapping_v2_config.json"
OUT_DIR = PROJECT / "results/crs_audit_v2"
OUT_JSON = OUT_DIR / "mapping_crs_audit_v2.json"
OUT_TXT = OUT_DIR / "mapping_crs_audit_v2.txt"
DEFAULT_BASENAME = "Nrb_sentinel2_extent3857.tif"

def collect_strings(x):
    vals = []
    if isinstance(x, dict):
        for v in x.values():
            vals.extend(collect_strings(v))
    elif isinstance(x, list):
        for v in x:
            vals.extend(collect_strings(v))
    elif isinstance(x, str):
        vals.append(x)
    return vals

def locate_raster(explicit=None):
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p.resolve(), ["explicit --raster path"]

    raster_strings = []
    if CONFIG_JSON.is_file():
        cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
        raster_strings = [
            s for s in collect_strings(cfg)
            if re.search(r"\.(tif|tiff)$", s, flags=re.I)
        ]

        for s in raster_strings:
            p = Path(s).expanduser()
            if p.is_file():
                return p.resolve(), [f"existing config path: {p}"]

    basenames = [Path(s.replace("\\", "/")).name for s in raster_strings]
    if DEFAULT_BASENAME not in basenames:
        basenames.append(DEFAULT_BASENAME)

    notes = []
    for name in dict.fromkeys(basenames):
        matches = [p for p in PROJECT.rglob(name) if p.is_file()]
        if len(matches) == 1:
            return matches[0].resolve(), [f"basename search: {matches[0]}"]
        if len(matches) > 1:
            notes.append(f"multiple matches for {name}: " + "; ".join(map(str, matches[:10])))

    raise FileNotFoundError(
        "Could not uniquely locate the source raster. Re-run with "
        "--raster /exact/path/to/source.tif\n"
        + "\n".join(notes)
        + ("\nConfig raster strings:\n" + "\n".join(raster_strings) if raster_strings else "")
    )

def parse_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    out = s.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if out.isna().any():
        raise ValueError("Could not parse boolean column")
    return out.astype(bool)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MAPPING_CSV.is_file():
        raise FileNotFoundError(MAPPING_CSV)

    raster_path, discovery = locate_raster(args.raster)

    df = pd.read_csv(MAPPING_CSV, low_memory=False)
    required = {
        "valid_cell", "is_padding_or_nodata",
        "xmin", "ymin", "xmax", "ymax"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping missing columns: {sorted(missing)}")

    valid = parse_bool(df["valid_cell"])
    pad = parse_bool(df["is_padding_or_nodata"])
    d = df.loc[valid & (~pad)].copy()

    widths = d["xmax"].astype(float) - d["xmin"].astype(float)
    heights = d["ymax"].astype(float) - d["ymin"].astype(float)

    mb = {
        "left": float(d["xmin"].min()),
        "bottom": float(d["ymin"].min()),
        "right": float(d["xmax"].max()),
        "top": float(d["ymax"].max()),
    }

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Source raster has no CRS metadata")

        crs = src.crs
        epsg = crs.to_epsg()
        rb = src.bounds
        px = abs(float(src.transform.a))
        py = abs(float(src.transform.e))

        contained = (
            mb["left"] >= rb.left - px and
            mb["right"] <= rb.right + px and
            mb["bottom"] >= rb.bottom - py and
            mb["top"] <= rb.top + py
        )

        cx = (mb["left"] + mb["right"]) / 2.0
        cy = (mb["bottom"] + mb["top"]) / 2.0
        xs = [mb["left"], cx, mb["right"]]
        ys = [mb["bottom"], cy, mb["top"]]
        lons, lats = rio_transform(crs, "EPSG:4326", xs, ys)

        audit = {
            "status": "PASSED" if contained else "REVIEW",
            "source_raster": str(raster_path),
            "source_raster_discovery": discovery,
            "crs_string": str(crs),
            "epsg": int(epsg) if epsg is not None else None,
            "is_projected": bool(crs.is_projected),
            "is_geographic": bool(crs.is_geographic),
            "linear_units": getattr(crs, "linear_units", None),
            "pixel_size_x": px,
            "pixel_size_y": py,
            "raster_bounds": {
                "left": float(rb.left), "bottom": float(rb.bottom),
                "right": float(rb.right), "top": float(rb.top),
            },
            "mapping_bounds": mb,
            "mapping_within_raster_1pixel_tolerance": bool(contained),
            "n_valid_nonpadding_cells": int(len(d)),
            "cell_width_unique": sorted(np.unique(np.round(widths, 9)).tolist()),
            "cell_height_unique": sorted(np.unique(np.round(heights, 9)).tolist()),
            "wgs84_sanity": {
                "lower_left_like": {"lon": float(lons[0]), "lat": float(lats[0])},
                "center": {"lon": float(lons[1]), "lat": float(lats[1])},
                "upper_right_like": {"lon": float(lons[2]), "lat": float(lats[2])},
            },
            "publication_rule": (
                "Use the CRS verified from the actual source-raster metadata. "
                "Do not infer EPSG from the raster filename alone."
            ),
        }

    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "=" * 88,
        "Nairobi v2 — Mapping CRS / Source Audit",
        "=" * 88,
        f"STATUS: {audit['status']}",
        f"Source raster: {audit['source_raster']}",
        f"CRS: {audit['crs_string']}",
        f"EPSG: {audit['epsg']}",
        f"Projected: {audit['is_projected']}",
        f"Linear units: {audit['linear_units']}",
        f"Raster pixel size: {audit['pixel_size_x']} x {audit['pixel_size_y']}",
        f"Valid non-padding cells: {audit['n_valid_nonpadding_cells']}",
        f"Cell widths: {audit['cell_width_unique']}",
        f"Cell heights: {audit['cell_height_unique']}",
        f"Mapping bounds: {audit['mapping_bounds']}",
        f"Raster bounds: {audit['raster_bounds']}",
        f"Mapping within raster (1-pixel tolerance): {audit['mapping_within_raster_1pixel_tolerance']}",
        f"WGS84 center: {audit['wgs84_sanity']['center']}",
        "",
        "Publication rule:",
        "Use the CRS reported by the actual raster metadata.",
        "Do not infer EPSG from the filename alone.",
        f"Audit JSON: {OUT_JSON}",
    ]
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    if audit["status"] != "PASSED":
        raise RuntimeError("CRS/source audit requires review.")

if __name__ == "__main__":
    main()
