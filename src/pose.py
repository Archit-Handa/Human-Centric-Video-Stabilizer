from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from numpy.typing import NDArray

from .utils import letterbox_resize, soft_argmax_2d

Keypoint = Tuple[float, float, float]   # (x, y, confidence)

class Pose2D:
    '''
    2D single-person pose via ONNX model (OpenCV DNN).
    If no model is available, provides a dummy 'mid-hip' near-center to keep the pipeline runnable.
    
    Args:
        onnx_path: Path to the ONNX model.
        input_size: Input size (width, height) expected by the ONNX model. Default is (192, 256).
    '''
    
    def __init__(
        self,
        onnx_path: Optional[str]=None,
        input_size: Tuple[int, int]=(192, 256)
    ) -> None:
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.net = None
        
        if self.onnx_path:
            try:
                self.net = cv2.dnn.readNetFromONNX(self.onnx_path)
            except Exception as e:
                print(f'[pose] Warning: Failed to load ONNX model: {e}. Using dummy pose instead.')
                
    def keypoints(self, frame_bgr: NDArray[np.uint8]) -> Dict[str, Keypoint]:
        '''
        Decode keypoints from the pose heatmaps, returning a small dict with a reference keypoint for stabilization.
        
        Args:
            frame_bgr: Input frame (H, W, 3) in uint8 BGR format.
            
        Returns:
            Dict of keypoints in (x, y, confidence) format. Preferred key: 'mid_hip'.
        '''
        
        H, W = frame_bgr.shape[:2]
        
        if self.net is None:
            cx, cy = W / 2.0, H * 0.55
            return {'mid_hip': (cx, cy, 1.0)}
        
        # Preprocess: letterbox to model input, then map coordinates back
        padded, scale, (pad_x, pad_y), (new_w, new_h) = letterbox_resize(frame_bgr, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]  # shape (1, 3, H, W)
        self.net.setInput(blob)
        hms = self.net.forward()    # shape (1, K, h, w)
        _, K, h, w = hms.shape
        hm = hms[0]
        
        points = []
        for k in range(K):
            m = hm[k]
            px_hm, py_hm = soft_argmax_2d(m)
            conf = float(m[int(round(py_hm)), int(round(px_hm))])
            
            # Map from heatmap coordinates to model input (padded) coordinates
            x_in = (px_hm + 0.5) * self.input_size[0] / w
            y_in = (py_hm + 0.5) * self.input_size[1] / H
            
            # Remove padding, then unscale to original
            x_un = x_in - pad_x
            y_un = y_in - pad_y
            x = x_un / scale
            y = y_un / scale
            points.append((float(x), float(y), conf))
        
        # Prefer COCO hips (indices 11, 12) if present/confident; else fallback
        if K > 13 and points[11][2] > 0.2 and points[12][2] > 0.2:
            mx = (points[11][0] + points[12][0]) / 2.0
            my = (points[11][1] + points[12][1]) / 2.0
            conf = (points[11][2] + points[12][2]) / 2.0
            return {'mid_hip': (mx, my, conf)}
        
        if K > 0:
            return {'nose': points[0]}
        
        cx, cy = W / 2.0, H / 2.0
        return {'center': (cx, cy, 0.0)}