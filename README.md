# Human-Centric Video Stabilization

The aim of this project is to process a video of a person walking in such a manner that the person stays at the center or a specified target point. The code achieves this by segmenting out the person from the background, running 2D pose detection to find a stable anchor (mid-hip, in this case), and stabilize the subject at the chosen target location. It also exports a side-by-side comparison video (Original vs Stabilized), pose keypoints, and stabilization transforms data along with the stabilized video.

> Outputs are written into a subfolder under `--outdir` named after the input video (without extension), e.g. `outputs/demo-video/`.

## Setup

### Requirements
- Python 3.10+
- OS: Linux/macOS/Windows (currently, the project works on CPU only)

### Install
```bash
# In repo root
python -m venv .venv

# For Linux or macOS
source .venv/bin/activate

# For Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### Project Layout
```
.
├─ src/
│  ├─ background.py
│  ├─ pose.py
│  ├─ rendering.py
│  ├─ stabilization.py
│  ├─ utils.py
│  └─ run.py
├─ models/
│  ├─ u2net_human_seg.onnx              # for human segmentation from background
│  └─ litehrnet_19_coco_256x192.onnx    # for 2D pose detection
├─ data/
│  └─ demo.mp4
└─ outputs/                             # created automatically
```

## Quick Start
The following commands are to be run in the repo root.

**Basic run**
```bash
python -m src.run \
  --input data/<demo_video>.mp4 \
  --outdir outputs \
  --seg-onnx models/<human_seg>.onnx \
  --pose-onnx models/<pose_2d>.onnx
```

**With alignment overlay & pose debug (recommended for inspection)**
```bash
python -m src.run \
  --input data/<demo_video>.mp4 \
  --outdir outputs \
  --seg-onnx models/<human_seg>.onnx \
  --pose-onnx models/<pose_2d>.onnx \
  --debug-mode
```

**Custom target point (for eg. lock the subject at x=640, y=360)**
```bash
python -m src.run \
  --input data/<demo_video>.mp4 \
  --outdir outputs \
  --seg-onnx models/<human_seg>.onnx \
  --pose-onnx models/<pose_2d>.onnx \
  --target-x 640 --target-y 360
```

**Keep cropped resolution (don't resize back to original size)**
```bash
python -m src.run \
  --input data/<demo_video>.mp4 \
  --outdir outputs \
  --seg-onnx models/<human_seg>.onnx \
  --pose-onnx models/<pose_2d>.onnx \
  --crop-no-resize
```

**Change comparison alignment (when not resizing the cropped stabilized pane)**
```bash
python -m src.run \
  --input data/<demo_video>.mp4 \
  --outdir outputs \
  --seg-onnx models/<human_seg>.onnx \
  --pose-onnx models/<pose_2d>.onnx \
  --crop-no-resize \
  --comp-v-align center --comp-h-align center
```

**Region of Interest (ROI) tuning (expand box, minimum area)**
```bash
python -m src.run \
  --input data/<demo_video>.mp4 \
  --outdir outputs \
  --seg-onnx models/<human_seg>.onnx \
  --pose-onnx models/<pose_2d>.onnx \
  --roi-scale 1.35 --roi-min-area 500
```

## Outputs
Inside `outputs/<video-name>/`:
- `stabilized.mp4` - stabilized video (cropped and optionally resized)
- `comparison.mp4` - side-by-side: Original (left) vs Stabilized (right)
  
  If `--debug-mode` is set, each pane also shows **target** (green), **reference** (red, eg. mid-hip), the **connecting line**, and **distance**.

- `pose_debug.mp4` - (only with `--debug-mode`) original video with keypoints/skeleton overlay
- `transforms.csv` - per-frame raw and smooth translations
- `pose_keypoints.json`/`pose_keypoints.csv` - per-frame joint `(x, y, conf)`

## Process
1. **Segmentation (BackgroundRemover)**

   Run a person segmentation ONNX model on each frame to get a binary mask.

2. **ROI selection**

   Take the largest contour, convert it to a bounding box, expand it (`--roi-scale`) and fit the pose model aspect (default **192x256** for pose).

3. **Pose (Pose2D, ROI)**

   Run a 2D pose ONNX model on the ROI. Decode joints by:
   - Upsampling heatmaps to the model input size,
   - Taking a discrete argmax per joint,
   - Computing a local peaky confidence score for the argmax.

4. **Reference Point**
   
   Prefer *mid_hip* (average of left and right hip joints) when confident; otherwise fall back to *nose* or ROI center.

5. **Stabilization**
   
   Compute per-frame translations that move the reference point to the target point (`--target-x/--target-y` or a preset via `--target`, currently supports 'center').
   Smooth with a moving average (`--smooth-window`).

6. **Uniform Crop**
   
   Compute a single crop that removes black borders for all frame after stabilization (`--crop auto`) or let the black borders stay (`--crop none`).
   Optionally, keep the cropped resolution (`--crop-no-resize`).

7. **Rendering**
   
   Save `stabilized.mp4` and `comparison.mp4`. The comparison video always labels panels **Original** / **Stabilized**; with `--debug-mode`, it also shows **target/ref points**, **line**, and **distance** on both panes.
   Pose overlays for debugging are saved to `pose_debug.mp4`.

## ONNX Models
Use any compatible ONNX pair that matches thed defaults:

- **Person Segmentation**
  - Expected input (default): **256x256** (`--seq-input`)
  - Output: single- or two-channel logits; converted to a person probability map.

- **2D Pose (single-person, COCO-17 joints recommended)**
  - Expected input (default): **192x256** (`--pose-input`)
  - Output: heatmaps shaped `(K, w, h)` with `K` joints.

Place your models in `models/` and point the flags to them:
```bash
--seg-onnx models/<human_seg>.onnx
--pose-onnx models/<pose_2d>.onnx
```

The program was developed and tested using the following open-source ONNX models:

- **Person Segmentation**: `u2net_human_seg.onnx` 
  - [Direct Download Link](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx)
  - **rembg** release asset
  
- **2D Pose**: `litehrnet_18_coco_256x192.onnx`
  - [GitHub Link](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/268_Lite-HRNet)
  - Run `download.sh` and choose the model
  > **Shape note:** Lite HRNet model name '256x192' follows HxW convention, while this project uses WxH format. Use `--pose-input 192x256` for this model.

## Flags (CLI)
|Flag|Type/Default|Purpose|
|---|---|---|
|`--input`, `-i`|**required**|Path to input video|
|`--output`, `-o`|Default: `outputs`|Root output directory (program creates a subfolder named after the video)|
|`--seg-onnx`|path|Person segmentation ONNX|
|`--pose-onnx`|path|2D pose ONNX|
|`--seg-input`|Default: `256x256`|Segmentation model input size (WxH)|
|`--pose-input`|Default: `192x256`|Pose model input size (WxH)|
|`--smooth-window`|Default: `11`|Moving average window (frames) for stabilization|
|`--target`|Default: `center`|Target preset (currently `center` locks near upper center)|
|`--target-x`/`--target-y`|Default: `None`|Override target position in pixels|
|`--crop`|Default: `auto`|Crop mode: `auto` computes a uniform crop; `none` disables cropping|
|`--crop-margin`|Default: `8`|Extra pixels cropped as safety around the computed uniform crop|
|`--crop-no-resize`|flag|If set, keep cropped size for outputs (don't resize back to original size)|
|`--comp-v-align`|Default: `center`|Vertical alignment for comparison when not resizing cropped pane; supports `top`, `center`, and `bottom`|
|`--comp-h-align`|Default: `center`|Horizontal alignment for comparison when not resizing cropped pane; supports `left`, `center`, and `right`|
|`--roi-scale`|Default: `1.25`|Expand the detected bbox before aspect fitting (adds margin; reduces jitter)|
|`--roi-min-area`|Default: `1500`|Minimum area in pixels to accept a person bbox|
|`--debug-mode`|flag|Enables alignment overlay on `comparison.mp4` and writes `pose_debug.mp4`|

## Runtime and Known Limitations

### Runtime

- **Test Hardware:** 2.3 GHz 8-core Intel Core i9 (9880H), 16 GB DDR4, AMD Radeon Pro 5500 4 GB + Intel UHD 630 1.5 GB

- **Execution Path:** OpenCV DNN on CPU (the discret GPU is not used by this project)

- **Expected Throughput** (default settings: `--seg-input 256x256`, `--pose-input 192x256`, `--roi-scale 1.25`, `--debug-mode` off):
  - **1080x1920 (1080p):** ~10-16 FPS
  > These are typical ranges for this CPU-only pipeline with lightweight ONNX models. Actual FPS varies with codec/IO, model architectures, and video content.

- **Tips to go faster:**
  - Use ROI defaults but smaller models (or smaller `--seg-input`/`--pose-input` if your ONNX supports it).
  - Keep `--debug-mode` off for production runs (overlays add a bit of work).
  - If needed, try a smaller output scale (run on 720p instead of 1080p).

### Known Limitations

- **CPU-only:** OpenCV DNN is used on CPU. GPU acceleration is not wired in this project yet.
  
- **Single-person assumption:** the largest-person mask is used; multi-person scenes aren’t disambiguated.
  
- **Occlusions / motion blur:** Heavy blur or long occlusions can degrade segmentation/pose, and may cause brief anchor drift.
  
- **ROI sensitivity:** If the segmentation box flickers, pose quality can wobble. The project mitigates this with `--roi-scale` (margin) and temporal smoothing, but extreme cases can still jitter.
  
- **Uniform crop reduces FOV:** Stabilization trims borders for all frames; aggressive motion could lead to larger crop (less field of view).

## Credits

**Author:** Archit Handa

**GitHub:** [@Archit-Handa](https://www.github.com/Archit-Handa)

*Project built with OpenCV + NumPy + tqdm and ONNX runtime via OpenCV DNN*