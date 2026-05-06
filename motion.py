from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple


@dataclass
class WaveResult:
    is_waving: bool
    confidence: float
    direction_changes: int
    avg_speed: float
    avg_horizontal_ratio: float


class WaveMotionTracker:
    def __init__(
        self,
        window_seconds: float,
        min_speed: float,
        min_dir_changes: int,
        min_move_px: float,
        min_horizontal_ratio: float,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_speed = min_speed
        self.min_dir_changes = min_dir_changes
        self.min_move_px = min_move_px
        self.min_horizontal_ratio = min_horizontal_ratio
        self._points: Deque[Tuple[float, float, float]] = deque()
        self.last_result = WaveResult(False, 0.0, 0, 0.0, 0.0)

    def update(self, x: float, y: float, timestamp: float) -> WaveResult:
        self._points.append((timestamp, x, y))
        self._prune(timestamp)
        if len(self._points) < 4:
            self.last_result = WaveResult(False, 0.0, 0, 0.0, 0.0)
            return self.last_result

        dir_changes, avg_speed, avg_ratio = self._analyze()
        confidence = self._score(dir_changes, avg_speed, avg_ratio)
        is_waving = (
            avg_speed >= self.min_speed
            and dir_changes >= self.min_dir_changes
            and avg_ratio >= self.min_horizontal_ratio
        )
        self.last_result = WaveResult(
            is_waving, confidence, dir_changes, avg_speed, avg_ratio
        )
        return self.last_result

    def get_points(self) -> List[Tuple[float, float]]:
        return [(p[1], p[2]) for p in self._points]

    def _prune(self, now: float) -> None:
        while self._points and now - self._points[0][0] > self.window_seconds:
            self._points.popleft()

    def _analyze(self) -> Tuple[int, float, float]:
        dir_changes = 0
        speeds: List[float] = []
        ratios: List[float] = []
        last_sign = 0

        for i in range(1, len(self._points)):
            t0, x0, y0 = self._points[i - 1]
            t1, x1, y1 = self._points[i]
            dt = max(t1 - t0, 1e-6)
            dx = x1 - x0
            dy = y1 - y0
            if abs(dx) < self.min_move_px and abs(dy) < self.min_move_px:
                continue
            speed = abs(dx) / dt
            speeds.append(speed)

            ratio = abs(dx) / (abs(dy) + 1e-6)
            ratios.append(ratio)

            sign = 1 if dx > 0 else -1
            if last_sign != 0 and sign != last_sign:
                dir_changes += 1
            last_sign = sign

        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        return dir_changes, avg_speed, avg_ratio

    def _score(self, dir_changes: int, avg_speed: float, avg_ratio: float) -> float:
        speed_score = min(avg_speed / max(self.min_speed, 1e-6), 1.5)
        change_score = min(dir_changes / max(self.min_dir_changes, 1), 1.5)
        ratio_score = min(avg_ratio / max(self.min_horizontal_ratio, 1e-6), 1.5)
        score = (speed_score + change_score + ratio_score) / 4.5
        return max(0.0, min(score, 1.0))
