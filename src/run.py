from __future__ import annotations

import argparse
import os
import csv
import numpy as np
from typing import Tuple

from .background import BackgroundRemover
from .pose import Pose2D
from .stabilization import target_point, compute_shifts, warp
from .rendering import open_video, writer, side_by_side

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Human-Centric Video Stabilization')
    
    parser.add_argument('--input', '-i', required=True, help='Path to the input video file.')
    parser.add_argument('--outdir', '-o', default='outputs', help='Path to the output directory.')
    
    parser.add_argument('--seg-onnx', dest='seg_onnx', default=None, help='Path to the person segmentation ONNX model.')
    parser.add_argument('--pose-onnx', dest='pose_onnx', default=None, help='Path to the 2D-pose estimation ONNX model.')
    
    parser.add_argument('--seg-input', default='256x256', help='Segmentation model input WxH (eg. "256x256").')
    parser.add_argument('--pose-input', default='192x256', help='Pose model input WxH (eg. "192x256").')
    
    parser.add_argument('--smooth-window', type=int, default=11, help='Moving average window (frames).')
    parser.add_argument('--target', default='center', help='Target point preset (eg. "center")')
    
    return parser.parse_args()

def _parse_wh(s: str) -> Tuple[int, int]:
    try:
        w_str, h_str = s.lower().split('x')
        return int(w_str), int(h_str)
    except Exception as e:
        raise argparse.ArgumentTypeError('Invalid WxH format, eg. "192x256"') from e

def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    cap, fps, W, H = open_video(args.input)
    seg_wh = _parse_wh(args.seg_input)
    pose_wh = _parse_wh(args.pose_input)
    
    bg = BackgroundRemover(args.seg_onnx, input_size=seg_wh)
    pose = Pose2D(args.pose_onnx, input_size=pose_wh)
    
    frames = []
    refs = []
    
    # 1. Collect reference points
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frames.append(frame)
        keypoints = pose.keypoints(frame)
        
        # Prefer ''mid_hip' if present; otherwise fallback to seg centroid; else frame center
        if 'mid_hip' in keypoints:
            x, y, _ = keypoints['mid_hip']
        else:
            mask = bg.segment(frame)
            if mask.any():
                ys, xs = np.nonzero(mask)
                x = float(xs.mean())
                y = float(ys.mean())
            else:
                x, y = float(W) / 2.0, float(H) / 2.0
        
        refs.append((float(x), float(y)))
        
    cap.release()
    
    cx, cy = target_point(W, H, args.target)
    dx_smooth, dy_smooth, dx_raw, dy_raw = compute_shifts(refs, cx, cy, args.smooth_window)
    
    # 2. Save transforms
    transforms_csv = os.path.join(args.outdir, 'transforms.csv')
    with open(transforms_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'dx_raw', 'dy_raw', 'dx_smooth', 'dy_smooth'])
        for i, (xr, yr, xs, ys) in enumerate(zip(dx_raw, dy_raw, dx_smooth, dy_smooth)):
            w.writerow([i, float(xr), float(yr), float(xs), float(ys)])
    
    # 3. Apply warps and write videos
    out_stab = writer(os.path.join(args.outdir, 'stabilized.mp4'), W, H, fps)
    out_comp = writer(os.path.join(args.outdir, 'comparison.mp4'), 2*W, H, fps)
    
    for i, frame in enumerate(frames):
        stabilized = warp(frame, float(dx_smooth[i]), float(dy_smooth[i]))
        composition = side_by_side(frame, stabilized)
        out_stab.write(stabilized)
        out_comp.write(composition)
    
    out_stab.release()
    out_comp.release()
    print(f'Done. Outputs in: {args.outdir}')
    
if __name__ == '__main__':
    main()