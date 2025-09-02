from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

def letterbox_resize(
    img: NDArray[np.uint8],
    dst_wh: Tuple[int, int]
) -> Tuple[NDArray[np.uint8], float, Tuple[int, int], Tuple[int, int]]:
    '''
    Resize an image using letterbox padding with aspect-ratio preserved and pad to dst_wh=(W, H).
    
    Args:
        img: Input image (H, W, 3) in uint8 format.
        dst_wh: (width, height) of the output image.
    
    Returns:
        padded_img: Padded image (H, W, 3) in uint8 format.
        scale: Scaling factor applied to the original image.
        (pad_x, pad_y): Top-left padding applied to the original image.
        (new_w, new_h): Resized (pre-pad) image size.
    '''
    
    Wt, Ht = dst_wh
    h, w = img.shape[:2]
    scale = min(Wt / w, Ht / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((Ht, Wt, 3), dtype=np.uint8)
    pad_x = (Wt - new_w) // 2
    pad_y = (Ht - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, float(scale), (pad_x, pad_y), (new_w, new_h)

def sigmoid(x: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x: NDArray, axis: int=1) -> NDArray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def peakiness_confidence(m_up: NDArray[np.float32], x: int, y: int, win: int = 9) -> float:
    '''
    Local confidence in a (win x win) window around the peak.
    
    Args:
        m_up: Upsampled heatmap (Ht, Wt).
        x: x coord of the peak.
        y: y coord of the peak.
        win: Window size (win x win).
    
    Returns:
        Confidence in the local peakiness.
    '''
    
    Ht, Wt = m_up.shape
    half = win // 2
    x0 = max(0, x - half)
    x1 = min(Wt, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(Ht, y + half + 1)
    patch = m_up[y0:y1, x0:x1]
    peak = float(m_up[y, x])
    
    # Exclude the center pixel from neighborhood mean
    mask = np.ones_like(patch, dtype=bool)
    mask[(y - y0), (x - x0)] = False
    neigh = float(patch[mask].mean()) if np.any(mask) else 0.0
    
    # Map to (0,1): higher if peak >> neighbors
    num = max(0.0, peak - neigh)
    den = (abs(peak) + abs(neigh) + 1e-6)
    
    return float(min(1.0, num / den))