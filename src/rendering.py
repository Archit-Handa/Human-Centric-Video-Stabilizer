from __future__ import annotations

import cv2
import numpy as np
from typing import Sequence, Tuple, Iterable
from numpy.typing import NDArray

# COCO-17 skeleton pairs (by joint index)
_COCO_PAIRS: list[tuple[int, int]] = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16),
]


def open_video(video_path: str) -> Tuple[cv2.VideoCapture, float, int, int]:
    '''
    Open a video file.
    
    Returns:
        (cap, fps, W, H): Video capture object, frame rate, width, and height.
    '''
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f'Failed to open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return (cap, fps, W, H)

def writer(output_path: str, W: int, H: int, fps: float) -> cv2.VideoWriter:
    '''
    Create an MP4 video writer with given size and frame rate.
    
    Args:
        output_path: Path to the output video file.
        W: Frame width in pixels.
        H: Frame height in pixels.
        fps: Frame rate in frames per second.
        
    Returns:
        Video writer object.
    '''
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (W, H))

def side_by_side(
    left_frame: NDArray[np.uint8],
    right_frame: NDArray[np.uint8],
    *,
    resize_right: bool=True,
    v_align: str='center',      # 'top'  | 'center' | 'bottom'
    h_align: str='center'       # 'left' | 'center' | 'right'
) -> NDArray[np.uint8]:
    '''
    Horizontally concatenate two frames. Resizes and aligns the right frame if necessary.
    
    Args:
        left_frame: Left frame (H, W, 3) in uint8 BGR format.
        right_frame: Right frame (H, W, 3) in uint8 BGR format (will be resized to left's size if needed).
        resize_right: If True, resize the right frame to the left frame's size. If False, keep the original size.
        v_align: Named preset for vertical alignment if resize_right=False. Currently supports 'center' (default) | 'top' | 'bottom'.
        h_align: Named preset for horizontal alignment if resize_right=False. Currently supports 'center' (default) | 'left' | 'right'.
    
    Returns:
        (H, 2W, 3) as the concatenated frame in uint8 BGR format.
    '''
    
    hl, wl = left_frame.shape[:2]
    hr, wr = right_frame.shape[:2]
    
    if resize_right:
        if (hl, wl) != (hr, wr):
            right_frame = cv2.resize(right_frame, (wl, hl), interpolation=cv2.INTER_LINEAR)
        return np.hstack([left_frame, right_frame])
    
    # No Resize: build a padded canvas
    target_h = max(hl, hr)
    target_w = max(wl, wr)
    comp = np.full((target_h, 2*target_w, 3), (0, 0, 0), dtype=np.uint8)
    
    def x_off(w: int) -> int:
        if h_align == 'left':
            return 0
        if h_align == 'right':
            return target_w - w
        return (target_w - w) // 2  # center
    
    def y_off(h: int) -> int:
        if v_align == 'top':
            return 0
        if v_align == 'bottom':
            return target_h - h
        return (target_h - h) // 2  # center

    yl = y_off(hl)
    xl = x_off(wl)
    yr = y_off(hr)
    xr = x_off(wr)
    
    comp[yl : yl + hl, xl : xl + wl] = left_frame
    comp[yr : yr + hr, target_w + xr : target_w +xr + wr] = right_frame

    return comp

def draw_keypoints(
    frame: NDArray[np.uint8],
    joints: Iterable[tuple[float, float, float]],
    *,
    conf_thr: float = 0.20,
    draw_skeleton: bool = True,
    draw_indices: bool = False,
) -> NDArray[np.uint8]:
    """
    Overlay keypoints (and optional skeleton) on a copy of the frame.

    Args:
        frame: BGR uint8 image.
        joints: iterable of (x,y,conf) in joint-index order.
        conf_thr: threshold below which a joint is considered missing.
        draw_skeleton: whether to draw COCO edges between visible joints.
        draw_indices: label each drawn joint with its index.
    """
    out = frame.copy()
    points = list(joints)
    H, W = out.shape[:2]

    # Skeleton
    if draw_skeleton:
        for a, b in _COCO_PAIRS:
            if a < len(points) and b < len(points):
                xa, ya, ca = points[a]
                xb, yb, cb = points[b]
                if ca >= conf_thr and cb >= conf_thr:
                    if 0 <= xa < W and 0 <= xb < W and 0 <= ya < H and 0 <= yb < H:
                        cv2.line(out, (int(round(xa)), int(round(ya))),
                                      (int(round(xb)), int(round(yb))),
                                      (0, 255, 255), 2, cv2.LINE_AA)

    # Joints
    for idx, (x, y, c) in enumerate(points):
        if c >= conf_thr and 0 <= x < W and 0 <= y < H:
            cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            if draw_indices:
                cv2.putText(out, str(idx), (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out

def draw_label(
    img: NDArray[np.uint8],
    text: str,
    corner: str='tl',   # tl | tr | bl | br
    pad: int=6
) -> NDArray[np.uint8]:
    out = img.copy()
    H, W = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    box_w = tw + pad * 2
    box_h = th + pad * 2
    
    if corner == 'tl':
        x0, y0 = 0, 0
    elif corner == 'tr':
        x0, y0 = W - box_w - 8, 8
    elif corner == 'bl':
        x0, y0 = 8, H - box_h - 8
    else:   # 'br'
        x0, y0 = W - box_w - 8, H - box_h - 8
        
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0.0, dst=out)
    
    tx = x0 + pad
    ty = y0 + box_h - pad - 2
    cv2.putText(out, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    
    return out

def draw_alignment(
    img: NDArray[np.uint8],
    ref_xy: Tuple[float, float],
    tgt_xy: Tuple[float, float],
    label_ref: str='ref',
    label_tgt: str='target',
    show_distance: bool=True
) -> NDArray[np.uint8]:
    out = img.copy()
    H, W = out.shape[:2]
    rx, ry = float(ref_xy[0]), float(ref_xy[1])
    tx, ty = float(tgt_xy[0]), float(tgt_xy[1])
    
    def _in(x: float, y: float) -> bool:
        return (0 <= x < W) and (0 <= y < H)
    
    if _in(rx, ry) and _in(tx, ty):
        cv2.line(out, (int(rx), int(ry)), (int(tx), int(ty)), (255, 255, 255), 2, cv2.LINE_AA)
    
    if _in(tx, ty):
        cv2.circle(out, (int(tx), int(ty)), 5, (0, 200, 0), -1, cv2.LINE_AA)
    if _in(rx, ry):
        cv2.circle(out, (int(rx), int(ry)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
    def _label(x: float, y: float, text: str, color: Tuple[int, int, int]) -> None:
        if not _in(x, y): return
        
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        pad = 4
        
        lx = int(min(max(0, x + 8), W - tw - 2))
        ly = int(min(max(th + 6, y - 8), H - 2))
        
        cv2.rectangle(out, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad), (0, 0, 0), -1)
        cv2.putText(out, text, (lx, ly), font, scale, color, thick, cv2.LINE_AA)
    
    _label(tx, ty, label_tgt, (0, 255, 0))
    _label(rx, ry, label_ref, (0, 0, 255))
    
    if show_distance and _in(rx, ry) and _in(tx, ty):
        d = float(np.hypot(rx - tx, ry - ty))
        d_text = f'dist = {d:.2f}px'

        mx = int((rx + tx) / 2.0)
        my = int((ry + ty) / 2.0) - 8
        mx = max(4, min(W - 4, mx))
        my = max(14, min(H - 4, my))
        cv2.putText(out, d_text, (mx, my), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    
    return out