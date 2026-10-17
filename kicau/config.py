from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DetectorConfig:
    mouth_distance_px: float = 60.0
    wave_speed_px_s: float = 75.0
    wave_dir_changes: int = 2
    wave_min_move_px: float = 4.0
    wave_window_s: float = 1.8
    wave_horizontal_ratio: float = 1.1
    mouth_hold_s: float = 0.25
    wave_hold_s: float = 0.1
    cooldown_s: float = 2.0
    tracking_grace_s: float = 0.3
    face_grace_s: float = 0.8


@dataclass(frozen=True)
class OverlayPlacement:
    left: bool = False
    top: bool = False
    mirror: bool = False
    center: bool = False


DEFAULT_OVERLAYS = (
    OverlayPlacement(),
    OverlayPlacement(left=True, mirror=True),
    OverlayPlacement(top=True),
    OverlayPlacement(top=True, left=True, mirror=True),
    OverlayPlacement(top=True, center=True),
    OverlayPlacement(top=True, center=True, left=True, mirror=True),
)


@dataclass(frozen=True)
class AppConfig:
    debug: bool = False
    camera_index: int = 0
    mirror_preview: bool = False
    dance_grace_s: float = 1.0
    test_duration_s: float = 5.0
    window_title: str = "Kicau Mania Detector"
    asset_dir: Path = PROJECT_ROOT / "assets"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    overlays: tuple[OverlayPlacement, ...] = DEFAULT_OVERLAYS

    @property
    def music_path(self) -> Path:
        return self.asset_dir / "music.mp3"

    @property
    def gif_path(self) -> Path:
        return self.asset_dir / "cat.gif"

