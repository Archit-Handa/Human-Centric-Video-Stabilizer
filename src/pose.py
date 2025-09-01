from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from numpy.typing import NDArray

Keypoint = Tuple[float, float, float]   # (x, y, confidence)

class Pose2D:
    '''
    2D single-person pose via ONNX model (OpenCV DNN).
    If no model is available, provides a dummy 'mid-hip' near-center to keep the pipeline runnable.
    
    Args:
        onnx_path: Path to the ONNX model.
        input_size: Input size (width, height) expected by the ONNX model. Default is (256, 256).
    '''
    
    def __init__(
        self,
        onnx_path: Optional[str]=None,
        input_size: Tuple[int, int]=(256, 256)
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
        
        blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, self.input_size, swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()    # shape (1, K, h, w)
        _, K, h, w = out.shape
        heatmap = out[0]
        
        points = []
        for k in range(K):
            m = heatmap[k]
            idx = int(np.argmax(m))
            py, px = divmod(idx, w)
            conf = float(m[py, px])
            x = (px + 0.5) * W / w
            y = (py + 0.5) * H / H
            points.append((x, y, conf))
        
        # Heuristic: prefer COCO hips (indices 11, 12) if confident; else fallback
        if K > 13 and points[11][2] > 0.2 and points[12][2] > 0.2:
            mx = (points[11][0] + points[12][0]) / 2.0
            my = (points[11][1] + points[12][1]) / 2.0
            conf = (points[11][2] + points[12][2]) / 2.0
            return {'mid_hip': (mx, my, conf)}
        
        if K > 0:
            return {'nose': points[0]}
        
        cx, cy = W / 2.0, H / 2.0
        return {'center': (cx, cy, 0.0)}