#!/usr/bin/env python3
"""Export browser assets for the ranked budget and the Evidence Lens.

For each scene this reorders the existing Spark-transcoded SPZ (degree-3 SH,
16 fractional bits) by opacity x projected area, so that any budget k is the
first k splats. SPZ v2 stores each attribute as a contiguous column, so the
permutation is exact: no Gaussian is re-quantized.

It also writes ``<scene>_evidence.bin``: one byte per Gaussian in the same
ranked order, holding the offline reliability proxy ``u_confidence`` from the
C6 checkpoint, quantized to 0..255.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path("/home/ushah/GSPLAT/stag_c6_clean_source/PromptDA/GSPLAT/work/"
              "fair_replica360_30k_b4/c6/models/pisr1000/seed_0")
SCENES = {
    "hotel_0": ("hotel_0_rand", "hotel_c6_full.spz"),
    "room_0": ("room_0_rand", "room_0_c6_full.spz"),
    "apartment_0": ("apartment_0_rand", "apartment_0_c6_full.spz"),
    "office_2": ("office_2_rand", "office_2_c6_full.spz"),
}
# Bytes per Gaussian for each SPZ v2 column, in file order.
COLUMNS = [("positions", 9), ("alphas", 1), ("colors", 3), ("scales", 3), ("rotations", 3)]


def sh_bytes(degree: int) -> int:
    return {0: 0, 1: 9, 2: 24, 3: 45}[degree]


def permute_spz(raw: bytes, order: np.ndarray) -> bytes:
    magic, version, n, degree, fractional, flags, _ = struct.unpack("<IIIBBBB", raw[:16])
    if magic != 0x5053474E or version != 2:
        raise ValueError(f"Unsupported SPZ header {magic:#x} v{version}")
    if len(order) != n:
        raise ValueError(f"Order has {len(order)} entries, SPZ has {n}")
    out = [raw[:16]]
    offset = 16
    for _, width in COLUMNS + [("sh", sh_bytes(degree))]:
        column = np.frombuffer(raw, np.uint8, n * width, offset).reshape(n, width)
        out.append(column[order].tobytes())
        offset += n * width
    if offset != len(raw):
        raise ValueError(f"Unexpected SPZ payload length {len(raw)} (parsed {offset})")
    return b"".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spz-dir", type=Path, default=ROOT / "demo/assets")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "publish/site/assets")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for key, (scene, spz_name) in SCENES.items():
        v = PlyData.read(str(MODELS / scene / "point_cloud/iteration_1000/point_cloud.ply"))["vertex"].data
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
        scales = np.sort(np.exp(np.column_stack([v[f"scale_{i}"] for i in range(3)]).astype(np.float64)), 1)
        score = opacity * scales[:, 2] * scales[:, 1]
        order = np.argsort(-score, kind="stable")
        confidence = np.asarray(v["u_confidence"], np.float64)

        raw = gzip.decompress((args.spz_dir / spz_name).read_bytes())
        # The SPZ must be in checkpoint order; check alpha against checkpoint opacity.
        n = struct.unpack("<I", raw[8:12])[0]
        alphas = np.frombuffer(raw, np.uint8, n, 16 + 9 * n).astype(np.float64)
        if np.abs(alphas - opacity * 255).max() > 1.01:
            raise ValueError(f"{spz_name} is not in checkpoint order")

        ranked_name = f"{key}_ranked.spz"
        ranked = gzip.compress(permute_spz(raw, order), compresslevel=9, mtime=0)
        (args.out_dir / ranked_name).write_bytes(ranked)

        evidence = np.clip(np.round(confidence[order] * 255), 0, 255).astype(np.uint8)
        evidence_name = f"{key}_evidence.bin"
        (args.out_dir / evidence_name).write_bytes(evidence.tobytes())

        manifest[key] = {
            "source_checkpoint": f"{scene}/point_cloud/iteration_1000/point_cloud.ply",
            "gaussians": int(n),
            "ranking": "descending sigmoid(opacity) x product of the two largest Gaussian scales",
            "ranked_spz": ranked_name,
            "ranked_spz_bytes": len(ranked),
            "ranked_spz_sha256": hashlib.sha256(ranked).hexdigest(),
            "evidence": evidence_name,
            "evidence_field": "u_confidence x 255",
            "confidence_percentiles": {str(p): float(np.percentile(confidence, p)) for p in (5, 25, 50, 75, 95)},
        }
        print(key, n, f"{len(ranked) / 2**20:.1f} MiB", f"evidence {len(evidence) / 2**20:.2f} MiB")

    (args.out_dir.parent.parent / "ranked_assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
