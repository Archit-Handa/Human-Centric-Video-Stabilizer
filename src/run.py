from __future__ import annotations

import argparse
import os
import csv
import cv2
import numpy as np
from typing import Tuple, Dict, List

from .background import BackgroundRemover
from .pose import Pose2D, Keypoint
from .stabilization import target_point, compute_shifts, warp
from .rendering import open_video, writer, side_by_side

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Human-Centric Video Stabilization')
    parser.add_argument('--input', '-i', required=True, help='Path to the input video file.')
    parser.add_argument('--outdir', '-o', default='outputs', help='Path to the output directory.')
    parser.add_argument('--seg-onnx', dest='seg_onnx', default=None, help='Path to the person segmentation ONNX model.')
    parser.add_argument('--pose-onnx', dest='pose_onnx', default=None, help='Path to the 2D-pose estimation ONNX model.')
    parser.add_argument('--smooth-window', type=int, default=11, help='Moving average window (frames).')
    parser.add_argument('--target', default='center', help='Target point preset (eg. "center")')
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    cap, fps, W, H = open_video(args.input)
    bg = BackgroundRemover(args.seg_onnx)
    pose = Pose2D(args.pose_onnx)
    
    frames = []
    refs = []
    
    # 1. Collect reference points
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frames.append(frame)
        keypoints = pose.keypoints(frame)
        if 'mid_hip' in keypoints:
            x, y, _ = keypoints['mid_hip']
        else:
            x, y, _ = next(iter(keypoints.values()))
        
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
    print(f'Done. Outputs in {args.outdir}')
    
if __name__ == '__main__':
    main()