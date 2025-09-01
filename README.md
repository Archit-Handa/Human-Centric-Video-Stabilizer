# Human-Centric Video Stabilization

Pipeline using only NumPy, OpenCV, PyTorch (for export), and ONNX.
- `src/` will contain modules for segmentation, pose, stabilization, and rendering.
- `models/` for ONNX files.
- `data/` for input videos.
- `outputs/` for results and diagnostics.

## Quick start
1) Create venv and install deps (see requirements.txt).
2) Place input video(s) in `data/`.
3) Run the pipeline (steps to follow).
