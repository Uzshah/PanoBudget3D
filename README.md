# PanoBudget3D

Interactive C6 Gaussian scene tours rendered with Spark and Three.js.

**[Open the public demo](https://uzshah.github.io/PanoBudget3D/)**

Choose Hotel 0, Room 0, Apartment 0, or Office 2, then select **Play guided tour**. Drag to look; W/A/S/D moves. Studio controls let you save the opening view, create camera stops, and export/import a tour JSON. Camera settings are saved separately for each scene in your browser. Export a JSON copy to move your tour to another browser. Saved local-server settings do not automatically transfer to this public site's origin.

## Scene assets

| Scene | Gaussians | Original PLY | Compressed SPZ (before ranking) |
| --- | ---: | ---: | ---: |
| Hotel 0 | 396,516 | 93.8 MiB | 9.8 MiB |
| Room 0 | 1,011,580 | 239.3 MiB | 27.3 MiB |
| Apartment 0 | 425,926 | 100.7 MiB | 10.6 MiB |
| Office 2 | 598,779 | 141.6 MiB | 14.7 MiB |

SPZ conversion retains every Gaussian and SH degree 3, with quantized parameters. It is lossy compression, not pruning. Compression uses Spark 2.3.0's `transcodeSpz`, `maxSh:3`, `fractionalBits:16`, and no opacity pruning. Original training checkpoints are not modified. See `compression.json` for the unranked sizes, counts, and checksums, and `ranked_assets.json` for the served ranked files. `tools/export_ranked_assets.py` reorders the SPZ columns without re-quantizing. Only the selected scene is downloaded.

## Run locally

```bash
python -m http.server 8765 --directory site
```

Open http://localhost:8765/. No build step or API key is needed. Rendering uses the visitor's GPU through WebGL2. Performance depends on the device; start with a desktop browser for the best experience.

## Camera settings from the local prototype

Browser storage is scoped to the site address. In the local viewer, select a scene and use **Export tour**; select the same scene on the public site and use **Import tour** to carry over your adjusted views.

## Deployment

The included GitHub Actions workflow publishes `site/` to GitHub Pages on a push to `main`. Configure repository Settings → Pages → Source as GitHub Actions. No external asset server or Git LFS is required.

## Research and attribution

These are existing C6-PI-SR reconstructions trained for 29k chart updates plus 1k ERP refinement. AI-assisted geometric priors and reliability guidance are part of the offline reconstruction. **Evidence lens:** each Gaussian carries the reliability proxy computed by the offline reconstruction (depth reprojection agreement, color agreement, cross-view support). The browser colors Gaussians by it and can hide those below a chosen confidence, in a Spark shader modifier. **Ranked budget:** assets are stored in descending opacity × area, so a k% budget draws the first k% of Gaussians. The ranking was selected on validation panoramas from the variants in `tools/budget_rank_eval.py` (including reliability-weighted ones, which did not improve quality); held-out test PSNR/SSIM/LPIPS are in `results/budget_rankings.json` and shown in the viewer. Spark's view-dependent LoD is offered for comparison and is not the manuscript's learned C5 gating. This viewer does not run AI inference in the browser. Active rendering budgets do not change the SPZ download size. Browser FPS is not a CUDA training or rendering benchmark.

Renderer: [Spark 2.3.0](https://sparkjs.dev/) and [Three.js 0.180.0](https://threejs.org/). Their MIT notices are included in `site/vendor/`. Scene reconstructions derive from the [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset); its research/noncommercial terms apply to scene assets. See `REPLICA_LICENSE.txt`.
