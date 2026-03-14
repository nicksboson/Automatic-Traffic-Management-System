import os
import time
import math
import numpy as np
import cv2
from ultralytics import YOLO

# --- Project Imports ---
from config import *
from detectors.empty_lane_detector import EmptyLaneDetector
from utils.draw_utils import draw_label, draw_traffic_light, draw_center_line
from utils.video_utils import resize_with_padding, is_on_left_side
from models.yolov8_manager import load_models

# ---------------- Initialize captures & models ----------------
caps = []
frame_widths = []
for p in VIDEO_FILES:
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        print(f"WARNING: Unable to open video {p}. Check path/codec.")
    caps.append(cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if width == 0:
        width = CELL_W
    frame_widths.append(width)

models = load_models()

if not os.path.isfile(TRACKER_YAML):
    print(f"WARNING: tracker yaml not found at '{TRACKER_YAML}'.")

# --------------- State ---------------
empty_detector = EmptyLaneDetector(EMPTY_THRESHOLD_SEC)
last_annotated = [np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8) for _ in range(NUM_ROADS)]
seen_ids_left = [set() for _ in range(NUM_ROADS)]
times_in_round = {r: 0 for r in range(1, NUM_ROADS+1)}
served_in_round = set()
current_green = None
next_yellow = None
prev_next_yellow = None

print("Starting traffic dashboard with LEFT SIDE detection only.")
print("Press ESC to exit.")


try:
    while True:
        # ---------- SCAN PHASE ----------
        seen_ids_left = [set() for _ in range(NUM_ROADS)]
        scan_start = time.time()

        while time.time() - scan_start < SCAN_SEC:
            for i in range(NUM_ROADS):
                cap = caps[i]
                ret, frame = cap.read()
                if not ret:
                    if LOOP_VIDEOS:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)

                draw_center_line(frame)

                model = models[i]
                annotated = frame.copy()
                vehicles_detected_left = False

                try:
                    if model is not None:
                        results = model.track(frame,
                                              tracker=TRACKER_YAML,
                                              persist=True,
                                              device=DEVICE,
                                              classes=CLASSES,
                                              conf=MIN_CONF,
                                              verbose=False)

                        if results and len(results) > 0:
                            r = results[0]
                            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                                for box in r.boxes:
                                    try:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    except Exception:
                                        continue
                                    bbox = [x1, y1, x2, y2]
                                    on_left = is_on_left_side(bbox, frame_widths[i])

                                    track_id = None
                                    if hasattr(box, "id") and box.id is not None:
                                        try:
                                            track_id = int(box.id[0].item())
                                        except:
                                            track_id = None

                                    conf = 0.0
                                    if hasattr(box, "conf") and box.conf is not None:
                                        try:
                                            conf = float(box.conf[0].item())
                                        except:
                                            conf = 0.0

                                    if on_left:
                                        if track_id is not None:
                                            seen_ids_left[i].add(track_id)
                                        vehicles_detected_left = True
                                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        label = f"ID:{track_id}" if track_id else f"Conf:{conf:.2f}"
                                        cv2.putText(annotated, label, (x1, y1-6),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    else:
                                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 180), 1)
                                        cv2.putText(annotated, "RIGHT", (x1, y1-6),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1)

                except Exception as e:
                    if VERBOSE:
                        print(f"[road{i+1}] tracking error: {e}")

                empty_detector.update(i, vehicles_detected_left)
                last_annotated[i] = annotated

            tiles = [resize_with_padding(f, CELL_W, CELL_H) for f in last_annotated]
            for tile in tiles:
                tile_h, tile_w = tile.shape[:2]
                cv2.line(tile, (tile_w//2, 0), (tile_w//2, tile_h), CENTER_LINE_COLOR, 1)

            top = np.hstack((tiles[0], tiles[1]))
            bottom = np.hstack((tiles[2], tiles[3]))
            grid = np.vstack((top, bottom))
            draw_label(grid, f"Scanning... {int(max(0, SCAN_SEC - (time.time()-scan_start)))}s left",
                       pos=(10, 25), bg_color=(50,50,50))
            cv2.imshow("Traffic Dashboard - Left Side Detection", grid)
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt

        counts = {i+1: len(seen_ids_left[i]) for i in range(NUM_ROADS)}
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Left side vehicle counts: {counts}")

        empty_roads = empty_detector.get_empty_roads()
        if empty_roads:
            print(f"[{ts}] Empty roads detected: {[r+1 for r in empty_roads]}")

        # ---------- SELECTION (with yellow light preview) ----------
        all_set = set(range(1, NUM_ROADS+1))
        blocked = set()
        if served_in_round != all_set:
            blocked = {r for r, cnt in times_in_round.items() if cnt >= 2}

        eligible = [r for r in range(1, NUM_ROADS+1) if r not in blocked]
        if not eligible:
            times_in_round = {r: 0 for r in range(1, NUM_ROADS+1)}
            served_in_round = set()
            eligible = list(range(1, NUM_ROADS+1))

        eligible_sorted = sorted(eligible, key=lambda r: counts.get(r, 0), reverse=True)

        # <<< CHANGED: prefer previously previewed yellow (if still eligible) so the preview matches next turn
        chosen = None
        if prev_next_yellow is not None and prev_next_yellow in eligible:
            chosen = prev_next_yellow
            # But if prev_next_yellow is not in the top slots, keep times/fairness consistent by removing it from eligible_sorted
            eligible_sorted = [r for r in eligible_sorted if r != chosen]
            print(f"[{ts}] Applying previous preview: promoting Road {chosen} to GREEN")
        else:
            if len(eligible_sorted) > 0:
                chosen = eligible_sorted[0]
                prev_next_yellow= eligible_sorted[1] if len(eligible_sorted) > 1 else None
            else:
                # fallback
                chosen = eligible[0]

        # Choose yellow light (next in line) from the remaining eligible list
        if len(eligible_sorted) > 0:
            next_yellow = eligible_sorted[0]  # top remaining
        else:
            next_yellow = None

        print(f"[{ts}] GREEN: Road {chosen}, YELLOW: Road {next_yellow if next_yellow else 'None'}")

        # Update fairness counters
        times_in_round[chosen] += 1
        served_in_round.add(chosen)
        if served_in_round == all_set:
            times_in_round = {r: 0 for r in range(1, NUM_ROADS+1)}
            served_in_round = set()

        # Save the previewed next_yellow so next cycle can respect it (unless blocked)
        prev_next_yellow = next_yellow  # <<< CHANGED: store preview for next selection

        # ---------- HOLD PHASE (with empty lane switching) ----------
        hold_start = time.time()
        switched_to_yellow = False

        while time.time() - hold_start < HOLD_SEC:
            # If current green becomes empty and next_yellow has vehicles, switch early
            if not switched_to_yellow and next_yellow is not None:
                empty_roads = empty_detector.get_empty_roads()
                if (chosen - 1) in empty_roads:
                    print(f"[{ts}] Switching from empty Road {chosen} to Road {next_yellow}")
                    chosen = next_yellow
                    next_yellow = None
                    switched_to_yellow = True
                    # update fairness for the new chosen
                    times_in_round[chosen] += 1
                    served_in_round.add(chosen)

            # Update live frame for chosen road only
            i = chosen - 1
            cap = caps[i]
            ret, frame = cap.read()
            if not ret:
                if LOOP_VIDEOS:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)

            draw_center_line(frame)

            model = models[i]
            annotated = frame.copy()
            vehicles_detected_left = False

            try:
                if model is not None:
                    results = model.track(frame,
                                          tracker=TRACKER_YAML,
                                          persist=True,
                                          device=DEVICE,
                                          classes=CLASSES,
                                          conf=MIN_CONF,
                                          verbose=False)

                    if results and len(results) > 0:
                        r = results[0]
                        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                            for box in r.boxes:
                                try:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                except:
                                    continue
                                bbox = [x1, y1, x2, y2]
                                on_left = is_on_left_side(bbox, frame_widths[i])

                                track_id = None
                                if hasattr(box, "id") and box.id is not None:
                                    try:
                                        track_id = int(box.id[0].item())
                                    except:
                                        track_id = None

                                conf = 0.0
                                if hasattr(box, "conf") and box.conf is not None:
                                    try:
                                        conf = float(box.conf[0].item())
                                    except:
                                        conf = 0.0

                                if on_left:
                                    vehicles_detected_left = True
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label = f"ID:{track_id}" if track_id else f"Conf:{conf:.2f}"
                                    cv2.putText(annotated, label, (x1, y1-6),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 180), 1)
                                    cv2.putText(annotated, "RIGHT", (x1, y1-6),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1)

            except Exception as e:
                if VERBOSE:
                    print(f"[road{chosen}] tracking error during hold: {e}")

            empty_detector.update(i, vehicles_detected_left)
            last_annotated[i] = annotated

            # Build display grid
            tiles = []
            for j in range(NUM_ROADS):
                tile = resize_with_padding(last_annotated[j], CELL_W, CELL_H)
                tile_h, tile_w = tile.shape[:2]
                cv2.line(tile, (tile_w//2, 0), (tile_w//2, tile_h), CENTER_LINE_COLOR, 1)

                empty_str = " [EMPTY]" if j in empty_detector.get_empty_roads() else ""
                draw_label(tile, f"Road {j+1}  Left:{counts[j+1]}{empty_str}",
                           pos=(10, 22), bg_color=(20,20,20))

                if (j+1) == chosen:
                    draw_traffic_light(tile, 'green')
                elif (j+1) == next_yellow:
                    draw_traffic_light(tile, 'yellow')
                else:
                    overlay = tile.copy()
                    cv2.rectangle(overlay, (0,0), (CELL_W, CELL_H), (0,0,100), -1)
                    tile = cv2.addWeighted(overlay, 0.15, tile, 0.85, 0)
                    draw_traffic_light(tile, 'red')

                tiles.append(tile)

            top = np.hstack((tiles[0], tiles[1]))
            bottom = np.hstack((tiles[2], tiles[3]))
            grid = np.vstack((top, bottom))

            status_text = f"GREEN: Road {chosen}"
            if next_yellow:
                status_text += f"  YELLOW: Road {next_yellow}"
            status_text += f"  (holding {int(max(0, HOLD_SEC - (time.time()-hold_start)))}s)"

            draw_label(grid, status_text, pos=(400, 320), bg_color=(30,30,30))
            draw_label(grid, "Detection: LEFT SIDE ONLY (Green=Counted, Red=Ignored)",
                       pos=(10, 710), bg_color=(40,40,40), text_color=CENTER_LINE_COLOR)

            cv2.imshow("Traffic Dashboard - Left Side Detection", grid)
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt

        current_green = chosen
        

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")

finally:
    for cap in caps:
        try:
            cap.release()
        except Exception:
            pass
    cv2.destroyAllWindows()
    print("Clean exit.")
