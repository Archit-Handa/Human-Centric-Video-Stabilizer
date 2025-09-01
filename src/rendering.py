from __future__ import annotations

import cv2
import numpy as np
from typing import Sequence, Tuple
from numpy.typing import NDArray

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
    comp = np.full((target_h, 2*target_w, 3), (0, 0, 0), dtype=left_frame.dtype)
    
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
    comp[yr : yr + hr, xr : xr + wr] = right_frame

    return comp