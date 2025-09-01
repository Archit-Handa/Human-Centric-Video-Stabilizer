from __future__ import annotations

import cv2
import numpy as np
from typing import Sequence, Tuple
from numpy.typing import NDArray

def target_point(W: int, H: int, mode: str='center') -> Tuple[float, float]:
    '''
    Choose a target anchor on the frame to lock the subject against.
    
    Args:
        W: Frame width in pixels.
        H: Frame height in pixels.
        mode: Named preset. Currently supports 'center' (default).
        
    Returns:
        (cx, cy): Target point on the frame, in (x, y) format for pixel coordinates.
    '''
    
    if mode == 'center':
        return (W / 2.0, H * 0.55)
    return (W / 2.0, H / 2.0)

def moving_average(x: Sequence[float], win: int=11) -> NDArray[np.float32]:
    '''
    Centered moving average smoothing.
    
    Args:
        x: Input sequence of numeric samples.
        win: Window size (odd number recommended).
        
    Returns:
        Smoothed sequence of numeric samples, same length as input.
    '''
    arr = np.asarray(x, dtype=np.float32)
    if win <= 1 or arr.size == 0:
        return arr
    
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(arr, kernel, mode='same').astype(np.float32)

def compute_shifts(
    refs: Sequence[Tuple[float, float]],
    cx: float,
    cy: float,
    smooth_win: int=11
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    '''
    Compute raw and smoothed x/y shifts that move each refrence point (cx, cy).
    
    Args:
        refs: Sequence of reference points per frame in (x, y) format.
        cx: Target x coordinate.
        cy: Target y coordinate.
        smooth_win: Moving average window size for smoothing.
        
    Returns:
        (dx_smooth, dy_smooth, dx_raw, dy_raw): Smoothed and raw x/y shifts, in (dx, dy) format.
    '''
    
    dx_raw = np.asarray([cx - x for (x, _) in refs], dtype=np.float32)
    dy_raw = np.asarray([cy - y for (_, y) in refs], dtype=np.float32)
    dx_smooth = moving_average(dx_raw, win=smooth_win)
    dy_smooth = moving_average(dy_raw, win=smooth_win)
    
    return dx_smooth, dy_smooth, dx_raw, dy_raw

def warp(frame_bgr: NDArray[np.uint8], dx: float, dy: float) -> NDArray[np.uint8]:
    '''
    Apply a pure-translation warp to a frame.
    
    Args:
        frame_bgr: Input frame (H, W, 3) in uint8 BGR format.
        dx: Horizontal translation in pixels along +x (rightwards).
        dy: Vertical translation in pixels along +y (downwards).
        
    Returns:
        Warped frame (H, W, 3) in uint8 BGR format.
    '''
    
    H, W = frame_bgr.shape[:2]
    M = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    out = cv2.warpAffine(
        src=frame_bgr,
        M=M,
        dsize=(W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return out

def compute_uniform_crop(
    W: int,
    H: int,
    dx: NDArray[np.float32],
    dy: NDArray[np.float32],
    margin: int=0
) -> Tuple[int, int, int, int]:
    '''
    Compute a single (x0, y0, w, h) crop that removes all black borders after applying the per-frame translations (dx, dy).
    
    Strategy:
        - For positive dx (shift right), black appears on the LEFT -> crop more on LEFT
        - For negative dx (shift left), black appears on the RIGHT -> crop more on RIGHT
        
        - For positive dy (shift down), black appears on the TOP -> crop more on TOP
        - For negative dy (shift up), black appears on the BOTTOM -> crop more on BOTTOM
    
    The crop is the intersection region that remains valid for all frames.
    A small margin (in pixels) is added to the crop to avoid edge artifacts for safety.
    
    Args:
        W: Frame width in pixels.
        H: Frame height in pixels.
        dx: Horizontal translation in pixels along +x (rightwards).
        dy: Vertical translation in pixels along +y (downwards).
        margin: Extra margin in pixels.
        
    Returns:
        (x0, y0, w, h): Crop region in (x0, y0, w, h) format.
    '''
    
    left = max(0, int(np.ceil(np.max(dx)))) + margin
    right = max(0, int(np.ceil(-np.min(dx)))) + margin
    top = max(0, int(np.ceil(np.max(dy)))) + margin
    bottom = max(0, int(np.ceil(-np.min(dy)))) + margin
    
    x0 = min(max(0, left), W)
    y0 = min(max(0, top), H)
    w = max(0, W - x0 - right)
    h = max(0, H - y0 - bottom)
    
    if w < 1 or h < 1:
        return 0, 0, W, H
    
    return x0, y0, w, h