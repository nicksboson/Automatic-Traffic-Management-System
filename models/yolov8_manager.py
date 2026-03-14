from ultralytics import YOLO
import os
from config import MODEL_PATH, NUM_ROADS, DEVICE, VERBOSE

def load_models():
    models = []
    for i in range(NUM_ROADS):
        try:
            model = YOLO(MODEL_PATH)
            models.append(model)
            if VERBOSE:
                print(f"[init] Loaded model for road {i+1} on device {DEVICE}")
        except Exception as e:
            print(f"[init] ERROR loading model {MODEL_PATH} for road {i+1}: {e}")
            models.append(None)
    return models
