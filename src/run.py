from __future__ import annotations

import argparse
import os
import csv
import cv2
import numpy as np
from typing import Tuple

from .background import BackgroundRemover
from .pose import Pose2D
from .stabilization import target_point, compute_shifts, warp, compute_uniform_crop
from .rendering import open_video, writer, side_by_side, draw_keypoints

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
    parser.add_argument('--target-x', type=float, default=None, help='Override target X (pixels).')
    parser.add_argument('--target-y', type=float, default=None, help='Override target Y (pixels).')
    
    parser.add_argument('--crop', choices=['auto', 'none'], default='auto', help='Border crop mode.')
    parser.add_argument('--crop-margin', type=int, default=8, help='Border crop margin (pixels) for safety.')
    
    parser.add_argument('--stab-no-resize', action='store_true', help='If set, stabilized.mp4 keeps cropped size instead of resizing back to original.')
    
    parser.add_argument('--comp-no-resize', action='store_true', help='If set, comparison.mp4 keeps cropped size of stabilized video instead of resizing back to original.')
    parser.add_argument('--comp-v-align', choices=['top', 'center', 'bottom'], default='center', help='Vertical alignment preset (eg. "center" | "top" | "bottom") when not resizing comparison.')
    parser.add_argument('--comp-h-align', choices=['left', 'center', 'right'], default='center', help='Horizontal alignment preset (eg. "center" | "left" | "right") when not resizing comparison.')
    
    parser.add_argument("--debug-pose", action="store_true",
               help="Write pose_debug.mp4 with joints/skeleton overlay")
    parser.add_argument("--pose-conf-thr", type=float, default=0.20,
                help="Confidence threshold for drawing joints")
    parser.add_argument("--debug-draw-skeleton", action="store_true",
                help="Draw COCO skeleton on debug video")
    parser.add_argument("--debug-draw-indices", action="store_true",
                help="Draw joint index labels on debug video")
    
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
    all_joints = []
    
    # Collect reference points
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frames.append(frame)
        keypoints = pose.keypoints(frame)
        
        if args.debug_pose:
            all_joints.append(pose.keypoints_all(frame))
        
        # Prefer 'mid_hip' if present; otherwise fallback to seg centroid; else frame center
        if 'mid_hip' in keypoints:
            x, y, _ = keypoints['mid_hip']
            # print('Midhip:', x, y)
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
    
    # Target Anchor
    if args.target_x is not None and args.target_y is not None:
        cx, cy = float(args.target_x), float(args.target_y)
    else:
        cx, cy = target_point(W, H, args.target)
    
    dx_smooth, dy_smooth, dx_raw, dy_raw = compute_shifts(refs, cx, cy, args.smooth_window)
    
    # Save transforms
    transforms_csv = os.path.join(args.outdir, 'transforms.csv')
    with open(transforms_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'dx_raw', 'dy_raw', 'dx_smooth', 'dy_smooth'])
        for i, (xr, yr, xs, ys) in enumerate(zip(dx_raw, dy_raw, dx_smooth, dy_smooth)):
            w.writerow([i, float(xr), float(yr), float(xs), float(ys)])
            
    # Decide crop box once for all frames
    if args.crop == 'auto':
        x0, y0, wc, hc = compute_uniform_crop(W, H, dx_smooth, dy_smooth, args.crop_margin)
    else:
        x0, y0, wc, hc = 0, 0, W, H
        
    # Size for stabilized.mp4
    if args.stab_no_resize:
        W_stab, H_stab = wc, hc
    else:
        W_stab, H_stab = W, H
    
    # Size for comparison.mp4
    W_comp, H_comp = 2*W, H
    
    # Apply warps and write videos
    out_stab = writer(os.path.join(args.outdir, 'stabilized.mp4'), W_stab, H_stab, fps)
    out_comp = writer(os.path.join(args.outdir, 'comparison.mp4'), W_comp, H_comp, fps)
    
    if args.debug_pose:
        out_dbg = writer(os.path.join(args.outdir, 'pose_debug.mp4'), W, H, fps)
    else:
        out_dbg = None
    
    for i, frame in enumerate(frames):
        stabilized = warp(frame, float(dx_smooth[i]), float(dy_smooth[i]))
        
        # Crop if requested
        if x0 != 0 or y0 != 0 or wc != W or hc != H:
            stabilized = stabilized[y0:y0+hc, x0:x0+wc]
            
        # Resize back (for stabilized.mp4) unless user opted out
        if not args.stab_no_resize and (stabilized.shape[1] != W or stabilized.shape[0] != H):
            stabilized = cv2.resize(stabilized, (W, H), interpolation=cv2.INTER_LINEAR)
        
        comparison = side_by_side(
            frame,
            stabilized,
            resize_right=not args.comp_no_resize,
            v_align=args.comp_v_align,
            h_align=args.comp_h_align
        )
        
        out_stab.write(stabilized)
        out_comp.write(comparison)
        
        if out_dbg is not None:
            dbg = draw_keypoints(
                frame,
                all_joints[i],
                conf_thr=args.pose_conf_thr,
                draw_skeleton=bool(args.debug_draw_skeleton),
                draw_indices=bool(args.debug_draw_indices),
            )
            out_dbg.write(dbg)
    
    out_stab.release()
    out_comp.release()
    
    if out_dbg is not None:
        out_dbg.release()
    
    print(f'Done. Outputs in: {args.outdir}')
    
if __name__ == '__main__':
    main()