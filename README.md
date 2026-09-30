# PanoBudget3D

Explore 3D rooms reconstructed from a few 360° photos, see where the reconstruction AI is confident, and choose how much of the scene your device draws.

**[Open the live tool](https://uzshah.github.io/PanoBudget3D/)**

## Features

- **Create from your own 360° photo:** choose **＋ Your 360° photo** (or **Try a sample**). Depth Anything V2 runs in the browser (WebGPU, or WebAssembly as a fallback) and turns the panorama into about 360k Gaussians you can walk into, in roughly 15 seconds. The photo never leaves your device. The evidence lens then shows where the estimated depth is unreliable, such as object edges.
- **Explore:** four indoor scenes (Hotel 0, Room 0, Apartment 0, Office 2) as Gaussian splats with full view-dependent color. Drag to look; W/A/S/D to move.
- **Evidence lens:** colors every Gaussian by the confidence the reconstruction AI gave it, from red (weak) to blue (strong). Press <kbd>L</kbd>. The **Hide below confidence** slider removes uncertain geometry live.
- **Render budget:** draw 10–100% of the Gaussians. The most visually important Gaussians are drawn first, so the room keeps its structure on low-end devices. The panel shows the measured image quality (PSNR) for each budget.
- **Camera tours:** play the guided tour, or build your own in **Studio controls** (save view, add stops, set duration) and export/import it as JSON.

## How it works

```text
360° photos ──► AI depth + poses ──► per-point confidence ──► Gaussian scene
                                                                  │
                     export: rank Gaussians by opacity × area,     │
                     store confidence (1 byte per Gaussian)  ◄─────┘
                                                                  │
browser (Spark + Three.js, WebGL2):                               ▼
  shader modifier ─► budget cut-off · confidence heatmap · confidence filter
```

Confidence combines depth reprojection agreement, color agreement and cross-view support, computed during reconstruction. Because the assets are stored in ranked order, a k% budget is just the first k% of Gaussians. It needs no extra download and runs entirely on the GPU.

### Measured quality (held-out 1024×512 panoramas, PSNR dB)

| Scene | 100% | 75% | 50% | 25% | 10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hotel 0 | 26.87 | 26.88 | 26.80 | 24.86 | 20.31 |
| Room 0 | 28.35 | 28.35 | 28.32 | 25.86 | 18.40 |
| Apartment 0 | 29.99 | 29.99 | 29.66 | 27.36 | 22.91 |
| Office 2 | 27.63 | 27.62 | 26.91 | 22.71 | 16.12 |

Half of the Gaussians keep quality within 0.7 dB of the full scene. Full metrics (SSIM, LPIPS) are in `results/budget_rankings.json`.

## Run locally

```bash
python -m http.server 8765 --directory site
```

Open http://localhost:8765/. No build step or API key is needed.

## Repository

| Path | Contents |
| --- | --- |
| `site/` | The browser tool (`spark-app.js`, `index.html`, `spark.css`), scene assets and measured quality |
| `site/create.js` | In-browser photo → depth → Gaussian pipeline |
| `tools/export_ranked_assets.py` | Ranks the Gaussians and writes the SPZ scene and confidence files |
| `tools/budget_rank_eval.py` | Measures held-out quality at each budget |
| `.github/workflows/pages.yml` | Deploys `site/` to GitHub Pages |

## Credits

Rendering: [Spark 2.3.0](https://sparkjs.dev/) and [Three.js 0.180.0](https://threejs.org/), MIT (notices in `site/vendor/`). Depth model: [Depth Anything V2 Small](https://huggingface.co/onnx-community/depth-anything-v2-small) (Apache-2.0) via [Transformers.js](https://github.com/huggingface/transformers.js). Scenes and the sample photo: [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset), research use only (see `REPLICA_LICENSE.txt`).
