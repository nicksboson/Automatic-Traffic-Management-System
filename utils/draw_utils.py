import cv2
import numpy as np
from config import CELL_W, CELL_H, CENTER_LINE_COLOR, CENTER_LINE_THICKNESS

def draw_label(img, text, pos=(10, 20), bg_color=(0,0,0), text_color=(255,255,255)):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x-4, y-18), (x+tw+4, y+6), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

def draw_traffic_light(img, state='red', center=(CELL_W - 36, 36)):
    radius = 16
    if state == 'green':
        cv2.circle(img, center, radius, (0, 255, 0), -1)
        cv2.circle(img, center, radius, (0, 200, 0), 2)
    elif state == 'yellow':
        cv2.circle(img, center, radius, (0, 255, 255), -1)
        cv2.circle(img, center, radius, (0, 200, 200), 2)
    else:
        cv2.circle(img, center, radius, (0, 0, 255), -1)
        cv2.circle(img, center, radius, (0, 0, 200), 2)

def draw_center_line(frame):
    if frame is None:
        return
    h, w = frame.shape[:2]
    center_x = w // 2
    cv2.line(frame, (center_x, 0), (center_x, h), CENTER_LINE_COLOR, CENTER_LINE_THICKNESS)
    cv2.putText(frame, "DETECT ZONE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CENTER_LINE_COLOR, 2)
