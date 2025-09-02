from __future__ import annotations

import argparse
import os
import csv
import cv2
import numpy as np
import json
from tqdm import tqdm
from typing import Tuple

from .background import BackgroundRemover
from .pose import Pose2D
from .stabilization import target_point, compute_shifts, warp, compute_uniform_crop
from .rendering import open_video, writer, side_by_side, draw_keypoints, draw_label, draw_alignment
from .utils import largest_bbox_from_mask, expand_and_fit_aspect

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
    parser.add_argument('--crop-no-resize', action='store_true', help='If set, stabilized and comparison videos keep cropped size instead of resizing back to original.')
    
    parser.add_argument('--comp-v-align', choices=['top', 'center', 'bottom'], default='center', help='Vertical alignment preset (eg. "center" | "top" | "bottom") when not resizing comparison.')
    parser.add_argument('--comp-h-align', choices=['left', 'center', 'right'], default='center', help='Horizontal alignment preset (eg. "center" | "left" | "right") when not resizing comparison.')
    
    parser.add_argument('--roi-scale', type=float, default=1.25, help='Scale factor to expand the person bbox before fitting pose aspect.')
    parser.add_argument('--roi-min-area', type=int, default=1500, help='Minimum mask area to accept a person bbox.')
    
    parser.add_argument("--debug-mode", action="store_true",
               help="Write pose_debug.mp4 with joints/skeleton overlay and comparison.mp4 with alignment overlay.")
    
    return parser.parse_args()

def _parse_wh(s: str) -> Tuple[int, int]:
    try:
        w_str, h_str = s.lower().split('x')
        return int(w_str), int(h_str)
    except Exception as e:
        raise argparse.ArgumentTypeError('Invalid WxH format, eg. "192x256"') from e

def main() -> None:
    args = parse_args()
    outdir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.input))[0])
    os.makedirs(outdir, exist_ok=True)
    print()
    
    try:
        cv2.setNumThreads(max(1, os.cpu_count() or 1))
    except Exception:
        pass
    
    cap, fps, W, H = open_video(args.input)
    seg_wh = _parse_wh(args.seg_input)
    pose_wh = _parse_wh(args.pose_input)
    
    bg = BackgroundRemover(args.seg_onnx, input_size=seg_wh)
    pose = Pose2D(args.pose_onnx, input_size=pose_wh)
    
    frames = []
    refs = []
    ref_labels = []
    all_joints = []
    pose_series = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    p1 = tqdm(total=(total_frames if total_frames > 0 else None), desc='Pass 1/2 [Stabilizing Video]', unit='frame')
    
    # Pass 1: Collect ROI-driven reference points
    last_bbox = None
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        idx += 1
        frames.append(frame)
        
        mask = bg.segment(frame)
        bbox = largest_bbox_from_mask(mask, min_area=args.roi_min_area)
        last_bbox = bbox
        
        if bbox is not None:
            bbox = expand_and_fit_aspect(bbox, W, H, aspect_w=pose_wh[0], aspect_h=pose_wh[1], scale=args.roi_scale)
            keypoints = pose.keypoints_roi(frame, bbox)
            joints_all = pose.keypoints_all_roi(frame, bbox)
            
            if 'mid_hip' in keypoints:
                x, y, _ = keypoints['mid_hip']
                name = 'mid_hip'
            else:
                name, (x, y, _) = next(iter(keypoints.items()))
        
            if args.debug_mode:
                all_joints.append(pose.keypoints_all_roi(frame, bbox))
        else:
            keypoints = pose.keypoints(frame)
            joints_all = pose.keypoints_all(frame)
            
            if 'mid_hip' in keypoints:
                x, y, _ = keypoints['mid_hip']
                name = 'mid_hip'
            else:
                x, y = float(W) / 2.0, float(H) / 2.0
                name = 'center'
            
            if args.debug_mode:
                all_joints.append(pose.keypoints_all(frame))
                
        refs.append((float(x), float(y)))
        ref_labels.append(name)
        pose_series.append(joints_all)
        
        if p1:
            p1.update(1)
        
    cap.release()
    if p1:
        p1.close()
    
    # Target Anchor
    if args.target_x is not None and args.target_y is not None:
        cx, cy = float(args.target_x), float(args.target_y)
    else:
        cx, cy = target_point(W, H, args.target)
    
    dx_smooth, dy_smooth, dx_raw, dy_raw = compute_shifts(refs, cx, cy, args.smooth_window)
    
    print('\nSaving Pose Keypoints and Stabilization Data...')
    
    # Save transforms (CSV)
    transforms_csv = os.path.join(outdir, 'transforms.csv')
    with open(transforms_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'dx_raw', 'dy_raw', 'dx_smooth', 'dy_smooth'])
        for i, (xr, yr, xs, ys) in enumerate(zip(dx_raw, dy_raw, dx_smooth, dy_smooth)):
            w.writerow([i, float(xr), float(yr), float(xs), float(ys)])
    print(f'- Saved transforms data at:     {outdir}/transforms.csv')
    
    # Save pose keypoints (JSON)
    kp_json_path = os.path.join(outdir, 'pose_keypoints.json')
    with open(kp_json_path, 'w') as f:
        json.dump({
            'joint_format': '(x, y, conf)',
            'frames': [[(float(x), float(y), float(c)) for (x, y, c) in fr] for fr in pose_series]
        }, f)
    print(f'- Saved pose keypoints data at: {outdir}/pose_keypoints.json')
    
    # Save pose keypoints (CSV)
    kp_csv_path = os.path.join(outdir, 'pose_keypoints.csv')
    with open(kp_csv_path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'joint_idx', 'x', 'y', 'conf'])
        for fi, fr in enumerate(pose_series):
            for ji, (x, y, c) in enumerate(fr):
                w.writerow([fi, ji, float(x), float(y), float(c)])
    print(f'- Saved pose keypoints data at: {outdir}/pose_keypoints.csv')
            
    # Decide crop box once for all frames
    if args.crop == 'auto':
        x0, y0, wc, hc = compute_uniform_crop(W, H, dx_smooth, dy_smooth, args.crop_margin)
    else:
        x0, y0, wc, hc = 0, 0, W, H
        
    # Size for stabilized.mp4
    if args.crop_no_resize:
        W_stab, H_stab = wc, hc
    else:
        W_stab, H_stab = W, H
    
    # Size for comparison.mp4
    W_comp, H_comp = 2*W, H
    
    # Writers
    out_stab = writer(os.path.join(outdir, 'stabilized.mp4'), W_stab, H_stab, fps)
    out_comp = writer(os.path.join(outdir, 'comparison.mp4'), W_comp, H_comp, fps)
    out_dbg = writer(os.path.join(outdir, 'pose_debug.mp4'), W, H, fps) if args.debug_mode else None
    
    print()
    p2 = tqdm(total=len(frames), desc='Pass 2/2 [Exporting Videos]', unit='frame')
    
    # Pass 2: Apply warps and write videos
    for i, frame in enumerate(frames):
        stabilized = warp(frame, float(dx_smooth[i]), float(dy_smooth[i]))
        
        # Crop if requested
        if x0 != 0 or y0 != 0 or wc != W or hc != H:
            stabilized = stabilized[y0:y0+hc, x0:x0+wc]
            
        right_h_pre, right_w_pre = stabilized.shape[:2]
            
        # Resize back (for stabilized.mp4) unless user opted out
        if not args.crop_no_resize and (stabilized.shape[1] != W or stabilized.shape[0] != H):
            stabilized = cv2.resize(stabilized, (W, H), interpolation=cv2.INTER_LINEAR)
        
        left_disp = draw_label(frame, 'Original', corner='tl')
        right_disp = draw_label(stabilized, 'Stabilized', corner='tl')
        
        if args.debug_mode:
            # Draw Left debug frame
            ref_left = refs[i]
            tgt_left = (cx, cy)
            
            left_disp = draw_alignment(left_disp, ref_left, tgt_left, label_ref=ref_labels[i], label_tgt='target')
            
            # Draw Right debug frame
            rx = refs[i][0] + float(dx_smooth[i]) - float(x0)
            ry = refs[i][1] + float(dy_smooth[i]) - float(y0)
            tx = float(cx) - float(x0)
            ty = float(cy) - float(y0)
            sx = sy = 1.0
            
            if not args.crop_no_resize and (right_w_pre != W or right_h_pre != H):
                sx = float(W) / float(right_w_pre)
                sy = float(H) / float(right_h_pre)
                
            rx *= sx; ry *= sy
            tx *= sx; ty *= sy
            
            right_disp = draw_alignment(right_disp, (rx, ry), (tx, ty), label_ref=ref_labels[i], label_tgt='target')
        
        comparison = side_by_side(
            left_disp,
            right_disp,
            resize_right=not args.crop_no_resize,
            v_align=args.comp_v_align,
            h_align=args.comp_h_align
        )
        
        out_stab.write(stabilized)
        out_comp.write(comparison)
        
        if out_dbg is not None:
            dbg = draw_keypoints(
                frame,
                all_joints[i],
                conf_thr=0.05,
                draw_skeleton=True,
                draw_indices=True,
            )
            out_dbg.write(dbg)
        
        if p2:
            p2.update(1)
    
    out_stab.release()
    out_comp.release()
    
    if out_dbg is not None:
        out_dbg.release()
        
    if p2:
        p2.close()
    
    print(f'\nDone. Outputs saved at: {outdir}')
    
if __name__ == '__main__':
    main()