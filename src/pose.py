from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
from numpy.typing import NDArray

from .utils import letterbox_resize, peakiness_confidence

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
    
    def _decode_from_heatmaps(
        self,
        hm: NDArray[np.float32],
        pad_x: int,
        pad_y: int,
        scale: float
    ) -> List[Keypoint]:
        '''
        For each channel:
            1) Upsample heatmap to (H_in, W_in) = self.input_size
            2) Argmax at input resolution (cv2.minMaxLoc)
            3) Confidence = local peakiness confidence at argmax
            4) Unpad and unscale to original frame coordinates
        '''
        
        W_in, H_in = self.input_size
        K, h, w = hm.shape
        points = []
        
        for k in range(K):
            m = hm[k]
            m_up = cv2.resize(m, (W_in, H_in), interpolation=cv2.INTER_CUBIC)
            _, _, _, maxLoc = cv2.minMaxLoc(m_up)
            x_in, y_in = maxLoc
            conf = peakiness_confidence(m_up, x_in, y_in, win=20)
            x_out = (x_in - pad_x) / scale
            y_out = (y_in - pad_y) / scale
            points.append((float(x_out), float(y_out), conf))
        
        return points
    
    def keypoints(self, frame_bgr: NDArray[np.uint8]) -> Dict[str, Keypoint]:
        '''
        Decode keypoints from the pose heatmaps, returning a small dict with a reference keypoint for stabilization.
        
        Args:
            frame_bgr: Input frame (H, W, 3) in uint8 BGR format.
            
        Returns:
            Dict of keypoints in (x, y, confidence) format. Preferred key: 'mid_hip'.
        '''
        
        points = self.keypoints_all(frame_bgr)
        K = len(points)
        
        # Prefer COCO hips (indices 11, 12) if present/confident; else fallback
        if K >= 13 and points[11][2] > 0.05 and points[12][2] > 0.05:
            mx = (points[11][0] + points[12][0]) / 2.0
            my = (points[11][1] + points[12][1]) / 2.0
            conf = (points[11][2] + points[12][2]) / 2.0
            return {'mid_hip': (mx, my, conf)}
        
        if K > 0:
            return {'nose': points[0]}
        
        H, W = frame_bgr.shape[:2]
        return {'center': (W / 2.0, H / 2.0, 0.0)}
    
    def keypoints_all(self, frame_bgr):
        H, W = frame_bgr[:2]
        
        if self.net is None:
            cx, cy = W / 2.0, H * 0.55
            return [(cx, cy, 1.0)]
        
        padded, scale, (pad_x, pad_y), (nw, nh) = letterbox_resize(frame_bgr, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1,3,H,W)
        self.net.setInput(blob)
        hms = self.net.forward()  # [1, K, h, w]
        hm = hms[0]
        
        return self._decode_from_heatmaps(hm, pad_x, pad_y, scale)