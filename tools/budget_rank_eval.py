#!/usr/bin/env python3
"""Compare static Gaussian rankings for top-k render budgets on held-out ERP views.

Each ranking orders a C6-PI-SR scene once, offline. A budget of k% renders the
first k% of that order, which is exactly what the browser viewer does after
the scene is exported in ranked order. Rankings are selected on the validation
panoramas and reported on the test panoramas.

Run inside the GSPLAT environment:
  source /home/ushah/GSPLAT/activate_gsplat.sh
  python /home/ushah/GSPLAT/web3d_budget_submission/tools/budget_rank_eval.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from native_gs.benchmark_c6pi_spherical_cuda import (
    load_graphdeco_ply, panorama_camera, quality, mean_dict,
)
from native_gs.spags_backend import SPaGSRasterizer

WORK = Path("/home/ushah/GSPLAT/stag_c6_clean_source/PromptDA/GSPLAT/work/fair_replica360_30k_b4/c6")
SCENES = ["hotel_0_rand", "room_0_rand", "apartment_0_rand", "office_2_rand"]
BUDGETS = [100, 75, 50, 25, 10]


def reliability_fields(ply: Path) -> dict[str, np.ndarray]:
    v = PlyData.read(str(ply))["vertex"].data
    log_scales = np.column_stack([v[f"scale_{i}"] for i in range(3)]).astype(np.float64)
    s = np.sort(np.exp(log_scales), axis=1)
    return {
        "opacity": 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64))),
        "area": s[:, 2] * s[:, 1],
        "confidence": np.asarray(v["u_confidence"], dtype=np.float64),
        "support": np.asarray(v["u_support"], dtype=np.float64),
        "depth_variance": np.asarray(v["u_depth_variance"], dtype=np.float64),
    }


def rankings(f: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """Return scores; higher is kept first."""
    footprint = f["opacity"] * np.sqrt(f["area"])
    rel = f["confidence"] * f["support"]
    conf = f["confidence"]
    low = conf < np.quantile(conf, 0.10)
    return {
        "random": np.random.default_rng(seed).random(len(footprint)),
        "opacity": f["opacity"],
        "footprint": footprint,
        "confidence": conf,
        "footprint_x_conf": footprint * conf,
        "footprint_x_conf^0.25": footprint * conf ** 0.25,
        "footprint_x_area": f["opacity"] * f["area"],
        "footprint_demote_low10": footprint * np.where(low, 0.25, 1.0),
        "footprint_x_1-depthvar": footprint / (1.0 + f["depth_variance"] / np.median(f["depth_variance"] + 1e-9)) ** 0.25,
    }


def cameras_for(scene: str, split: str, width: int):
    source = WORK / "prepared" / scene
    metadata = json.loads((source / "metadata/cameras.json").read_text())
    faces = json.loads((source / "metadata/faces.json").read_text())
    prep = json.loads((source / "prep_summary.json").read_text())
    raw = Path(prep["source_scene"])
    face_zero = np.asarray(faces[0]["tile_rotation"], dtype=np.float32)
    out = []
    for pano in prep["split_panoramas"][split]:
        entries = sorted((e for e in metadata if e["split"] == split and str(e["pano_id"]) == pano),
                         key=lambda e: int(e["face_index"]))
        target = np.asarray(Image.open(raw / f"{pano}_rgb_pano.jpg").convert("RGB").resize(
            (width, width // 2), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        out.append((pano, panorama_camera(entries[0], face_zero, width),
                    torch.from_numpy(target).permute(2, 0, 1).cuda()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--erp-width", type=int, default=1024)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "results" / "budget_rankings.json")
    args = parser.parse_args()

    rasterizer = SPaGSRasterizer().cuda()
    results = {"format": "panobudget_rank_eval_v1", "erp_resolution": [args.erp_width, args.erp_width // 2],
               "budgets_percent": BUDGETS, "scenes": {}}
    for scene in args.scenes:
        ply = WORK / "models/pisr1000/seed_0" / scene / "point_cloud/iteration_1000/point_cloud.ply"
        params = load_graphdeco_ply(ply, torch.device("cuda"))
        scores = rankings(reliability_fields(ply))
        n = params["positions"].shape[0]
        scene_out = {"gaussians": int(n)}
        for split in ("val", "test"):
            cams = cameras_for(scene, split, args.erp_width)
            split_out = {}
            for name, score in scores.items():
                order = torch.from_numpy(np.argsort(-score, kind="stable")).cuda()
                per_budget = {}
                for budget in BUDGETS:
                    if budget == 100 and name != "random":
                        continue
                    keep = order[: max(1, n * budget // 100)]
                    sub = {k: v[keep].contiguous() for k, v in params.items()}
                    items = []
                    with torch.inference_mode():
                        for pano, cam, target in cams:
                            img, _ = rasterizer.render(**sub, mode=2, K=16, active_sh_bases=16,
                                                       scale_modifier=1.0, camera=cam, to_chw=True,
                                                       use_median_depth=False)
                            items.append(quality(img.clamp(0, 1), target))
                    per_budget[str(budget)] = mean_dict(items)
                split_out[name] = per_budget
            full = split_out["random"].pop("100")
            split_out["full"] = full
            scene_out[split] = split_out
            print(scene, split, "full", round(full["psnr"], 2), {
                k: [round(v[str(b)]["psnr"], 2) for b in BUDGETS[1:]] for k, v in split_out.items() if k != "full"})
        results["scenes"][scene] = scene_out
        del params
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
