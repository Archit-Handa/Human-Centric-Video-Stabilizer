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

def side_by_side(left_frame: NDArray[np.uint8], right_frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    '''
    Horizontally concatenate two frames. Resizes the right frame if necessary.
    
    Args:
        left_frame: Left frame (H, W, 3) in uint8 BGR format.
        right_frame: Right frame (H, W, 3) in uint8 BGR format (will be resized to left's size if needed).
    
    Returns:
        (H, 2W, 3) as the concatenated frame in uint8 BGR format.
    '''
    
    hl, wl = left_frame.shape[:2]
    hr, wr = right_frame.shape[:2]
    
    if (hl, wl) != (hr, wr):
        right_frame = cv2.resize(right_frame, (wl, hl))
        
    return np.hstack([left_frame, right_frame])