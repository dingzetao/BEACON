# BEACON Inference GUI (NiceGUI)

Research workbench for **local / single-server** use: H&E image(s) → dense spatial abundance for **Tumor_EGFR** and **Mac_EREG** (UNI + GAT).

## Input recommendation (important)

**Use a file or directory path on the compute server — do not upload full multi-hundred-MB WSIs through the browser.**


| Approach                                         | When to use                                                                 |
| ------------------------------------------------ | --------------------------------------------------------------------------- |
| **Path to image / core directory** (recommended) | Real TMA cores or H&E `.tif/.jpg` already on disk (same machine as the GUI) |
| Browser upload of whole WSI                      | **Not supported in v1** — slow, memory-heavy, fragile on shared nodes       |


Supported extensions (OpenCV / `cv2.imread`): `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`.  
For `.svs` / `.ndpi`, first dearray cores with `WSI/TMA_IHC_dearray.py` (or equivalent), then point the GUI at `WSI/output_cores` or a single core path.

## What this wraps

From the BEACON repo root (a `BEACON` → `.` symlink keeps package imports working):

- Package: `beacon/infer.py` → `infer_dense_heatmap` / `save_dense_heatmap`
- Checkpoints: set in the UI (lab defaults may point at `decoder/gat_seed/...`; weights are **not** shipped in this repo)
- Same dense sliding-window recipe as `scripts/05_infer_tma.py`



## Environment

Use the same env as BEACON inference (e.g. `trident`), which already has PyTorch, TRIDENT/UNI, torch-geometric, opencv, etc. Then:

```bash
pip install -r requirements-gui.txt
# or: pip install nicegui
```

Run on a **compute node with GPU** if possible (UNI encoding is heavy). Do not assume the login node.

## Start

**GUI 不会自动切换 conda 环境**——用哪个 `python` 启动，就用哪个环境里的包。

```bash
cd /path/to/BEACON   # this repository root

# Use the Python env that has PyTorch + TRIDENT/UNI
python -c "from trident.patch_encoder_models import encoder_factory; print('TRIDENT OK')"
python -m gui
```

Default input directory: `gui/WSI_data/` (gitignored; create locally or point the UI at any on-disk path).

Open `http://localhost:8088` after an SSH tunnel to the GPU node, e.g.:

```text
ssh -L 8088:<gpu-node>:8088 <user>@<login-host>
```

### If you see `No module named 'trident'`

1. **Wrong Python** — start the GUI with the env that has TRIDENT installed.
2. **Broken install** — reinstall [TRIDENT](https://github.com/mahmoodlab/TRIDENT) in that env (`pip install -e .`), then re-test `encoder_factory` before launching the GUI.

## Workflow

1. Set **input mode**: single image path, or directory of cores (default: `gui/WSI_data/`).
2. Confirm checkpoints for Tumor_EGFR / Mac_EREG.
3. Set device (`auto` / `cuda` / `cpu`), `batch_size`, `patch_size`, `stride`, `k_neighbors`.
4. Click **Run BEACON inference**. Progress and logs refresh every ~0.5 s.
5. When done, view heatmaps in the Results panel; files are under `gui_runs/run_YYYYMMDD_HHMMSS/`.



## Outputs

Per image / niche:


| File                  | Meaning                               |
| --------------------- | ------------------------------------- |
| `{Tumor_EGFR          | Mac_EREG}.npy`                        |
| `{Tumor_EGFR          | Mac_EREG}_dense_heatmap.png/.pdf`     |
| `input_thumbnail.jpg` | Downscaled preview of input           |
| `gallery.png`         | Side-by-side niches (if both ran)     |
| `summary.csv`         | mean abundance + patch counts         |
| `run_config.json`     | Snapshot of paths and hyperparameters |




## Known limits (v1)

- No QuPath-style multi-resolution WSI viewer.
- No browser upload of giant slides; path-only.
- No multi-user auth.
- No training UI (use existing scripts).
- Full-slide gigapixel WSIs may OOM if loaded as one OpenCV image — prefer **core / ROI tiles**.
- Requires TRIDENT + UNI for the encoder; missing deps / CUDA OOM appear in the log + notify.



## Architecture

```
gui/
  main.py      # NiceGUI pages + timer refresh
  state.py     # ConfigState + RunState
  adapter.py   # thin call into BEACON/beacon (no copied training logic)
  defaults.py  # project-relative checkpoint / output defaults
```

Long jobs run in a **background thread**; the UI only polls state.