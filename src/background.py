from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray

class BackgroundRemover:
    '''
    Person/background matting using an ONNX model loaded via OpenCV DNN.
    If no model is provided or loading fails, returns a full-frame mask to keep the pipeline runnable.
    
    Args:
        onnx_path: Path to the ONNX model.
        input_size: Input size (width, height) expected by the ONNX model. Default is (256, 256).
        threshold: Probability threshold for the person/background segmentation. Default is 0.5.
    '''
    
    def __init__(
        self,
        onnx_path: Optional[str]=None,
        input_size: Tuple[int, int]=(256, 256),
        threshold: float=0.5
    ) -> None:
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.threshold = threshold 
        self.net = None
        
        if self.onnx_path:
            try:
                self.net = cv2.dnn.readNetFromONNX(self.onnx_path)
            except Exception as e:
                print(f'[background] Warning: Failed to load ONNX model: {e}. Using dummy mask instead.')
    
    def segment(self, frame_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        '''
        Compute a binary person mask for a BGR frame.
        
        Args:
            frame_bgr: Input frame (H, W, 3) in uint8 BGR format.
        
        Returns:
            mask: Binary mask (H, W) in uint8 where 0 indicates background and 1 indicates person.
        '''
        H, W = frame_bgr.shape[:2]
        
        # If the model is not loaded, return a dummy mask where whole frame is treated as a "person" to keep the pipeline flowing
        if self.net is None:
            return np.ones((H, W), dtype=np.uint8)
        
        blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, self.input_size, swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()    # shape (1, C, H, W)
        
        if out.shape[1] == 1:
            prob = 1.0 / (1.0 + np.exp(-out[0, 0]))
        else:
            e = np.exp(out - out.max(axis=1, keepdims=True))
            prob = (e[:, 1] / e.sum(axis=1))[0]
            
        mask = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = (mask > self.threshold).astype(np.uint8)
        
        return mask