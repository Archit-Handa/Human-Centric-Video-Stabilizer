from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple
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

def largest_bbox_from_mask(mask: NDArray[np.uint8], min_area: int=500) -> Optional[Tuple[int, int, int, int]]:
    '''
    Find the largest connected component bounding box (x0, y0, x1, y1) from a binary mask.
    
    Args:
        mask: Binary mask (H, W).
        min_area: Minimum area of the largest connected component.
    
    Returns:
        (x0, y0, x1, y1): Bounding box of the largest connected component.
    '''
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < float(min_area):
        return None
    
    x0, y0, w, h = cv2.boundingRect(best)
    return (int(x0), int(y0), int(x0 + w), int(y0 + h))

def expand_and_fit_aspect(
    box: Tuple[int, int, int, int],
    W: int, H: int,
    aspect_w: int, aspect_h: int,
    scale: float=1.25
) -> Tuple[int, int, int, int]:
    '''
    Expand a bounding box to fit the aspect ratio of the original image and clamp to image bounds.
    
    Args:
        box: Bounding box (x0, y0, x1, y1).
        W: Original image width.
        H: Original image height.
        aspect_w: Aspect ratio width.
        aspect_h: Aspect ratio height.
        scale: Scale factor.
    
    Returns:
        (x0, y0, x1, y1): Expanded bounding box.
    '''
    
    x0, y0, x1, y1 = map(int, box)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0) * scale
    h = (y1 - y0) * scale
    
    target_aspect_ratio = aspect_w / aspect_h
    current_aspect_ratio = w / h if h > 1e-6 else target_aspect_ratio
    
    if current_aspect_ratio > target_aspect_ratio:
        h = w / target_aspect_ratio
    else:
        w = h * target_aspect_ratio
    
    x0n = int(round(cx - w / 2.0))
    y0n = int(round(cy - h / 2.0))
    x1n = int(round(cx + w / 2.0))
    y1n = int(round(cy + h / 2.0))
    
    x0n = max(0, min(x0n, W - 1))
    y0n = max(0, min(y0n, H - 1))
    x1n = max(x0n + 1, min(x1n, W))
    y1n = max(y0n + 1, min(y1n, H))
    
    return x0n, y0n, x1n, y1n