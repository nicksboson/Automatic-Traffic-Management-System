import cv2
import numpy as np
from config import CELL_W, CELL_H

def resize_with_padding(frame, target_w=CELL_W, target_h=CELL_H):
    if frame is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - nw) // 2
    y_off = (target_h - nh) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas

def is_on_left_side(bbox, frame_width):
    x_center = (bbox[0] + bbox[2]) / 2.0
    return x_center < (frame_width / 2.0)
