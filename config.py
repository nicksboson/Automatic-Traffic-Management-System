# Configuration and constants

VIDEO_FILES = ["road4.mp4", "road9.mp4", "road3.mp4", "road1.mp4"]
MODEL_PATH = "yolov8n.pt"
TRACKER_YAML = "bytetrack.yaml"
SCAN_SEC = 1.0
HOLD_SEC = 5.0
EMPTY_THRESHOLD_SEC = 3.0
CELL_W, CELL_H = 620, 340
MIN_CONF = 0.01
CLASSES = [0, 2, 3, 5, 7]
LOOP_VIDEOS = True
VERBOSE = False

DETECT_LEFT_ONLY = True
CENTER_LINE_COLOR = (255, 255, 0)
CENTER_LINE_THICKNESS = 2

import torch, os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ROADS = 4

assert len(VIDEO_FILES) == NUM_ROADS, "Provide exactly 4 video paths in VIDEO_FILES"
