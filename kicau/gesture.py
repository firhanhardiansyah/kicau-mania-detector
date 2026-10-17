from typing import Optional, Tuple

from .config import DetectorConfig
from .motion import WaveResult


class GestureRecognizer:
    """Recognize a held mouth-covering pose followed by a wave, with dropout grace."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.state = "IDLE"
        self.mouth_start: Optional[float] = None
        self.wave_start: Optional[float] = None
        self.cooldown_until = 0.0
        self._mouth_seen_at = float("-inf")
        self._wave_active_at = float("-inf")

    def update(
        self, mouth_covered: bool, waving_hand_present: bool, wave: WaveResult, now: float
    ) -> Tuple[bool, str]:
        if mouth_covered:
            self._mouth_seen_at = now
        mouth_recent = now - self._mouth_seen_at <= self.config.tracking_grace_s
        if waving_hand_present and wave.is_waving:
            self._wave_active_at = now
        wave_active = now - self._wave_active_at <= self.config.tracking_grace_s
        triggered = False
        display_state = self.state

        if self.state == "IDLE":
            if mouth_recent:
                self.state = "MOUTH_COVERED"
                self.mouth_start = now
                display_state = self.state
        elif self.state == "MOUTH_COVERED":
            if not mouth_recent:
                self.state = "IDLE"
                self.mouth_start = None
            elif (
                self.mouth_start is not None
                and now - self.mouth_start >= self.config.mouth_hold_s
            ):
                self.state = "WAVING"
                self.wave_start = None
                display_state = "MOUTH_COVERED"
        elif self.state == "WAVING":
            if not mouth_recent:
                self.state = "IDLE"
                self.wave_start = None
            else:
                if wave_active:
                    if self.wave_start is None:
                        self.wave_start = now
                    elif now - self.wave_start >= self.config.wave_hold_s:
                        triggered = True
                        display_state = "DANCE_DETECTED"
                        self.state = "COOLDOWN"
                        self.cooldown_until = now + self.config.cooldown_s
                        self.wave_start = None
                else:
                    display_state = "MOUTH_COVERED"
                    self.wave_start = None
        elif self.state == "COOLDOWN":
            if now >= self.cooldown_until:
                self.state = "IDLE"
                self.mouth_start = None
                self.wave_start = None

        if not triggered and self.state == "IDLE":
            display_state = "IDLE"
        return triggered, display_state

