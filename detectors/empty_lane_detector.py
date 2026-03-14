import time
from config import NUM_ROADS, EMPTY_THRESHOLD_SEC

class EmptyLaneDetector:
    """Detect if a lane has been empty for a threshold duration"""
    def __init__(self, threshold_sec=EMPTY_THRESHOLD_SEC):
        self.threshold_sec = threshold_sec
        self.last_detection_time = {}
        self.is_empty = {}
        for i in range(NUM_ROADS):
            self.last_detection_time[i] = time.time()
            self.is_empty[i] = False

    def update(self, road_id, has_vehicles):
        current_time = time.time()
        if has_vehicles:
            self.last_detection_time[road_id] = current_time
            self.is_empty[road_id] = False
        else:
            time_empty = current_time - self.last_detection_time.get(road_id, current_time)
            self.is_empty[road_id] = time_empty >= self.threshold_sec

    def get_empty_roads(self):
        return [i for i, empty in self.is_empty.items() if empty]
