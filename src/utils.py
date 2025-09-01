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

def soft_argmax_2d(hm: NDArray) -> Tuple[float, float]:
    '''
    Sub-pixel peak estimate from a single heatmap (h, w) using soft-argmax.
    
    Args:
        hm: Heatmap (h, w).
    
    Returns:
        (x, y): Sub-pixel peak estimate coordinate in (x, y) format.
    '''
    
    h, w = hm.shape
    p = hm - hm.max()
    p = np.exp(p)
    z = p.sum()
    
    if z <= 0:
        idx = int(np.argmax(hm))
        py, px = divmod(idx, w)
        return float(px), float(py)
    
    p /= z
    xs = (p.sum(axis=0) * np.arange(w)).sum()
    ys = (p.sum(axis=1) * np.arange(h)).sum()
    return float(xs), float(ys)