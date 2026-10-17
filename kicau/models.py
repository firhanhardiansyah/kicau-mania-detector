"""Download and cache the MediaPipe model files independently of detection."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request

import certifi

from .config import PROJECT_ROOT


@dataclass(frozen=True)
class ModelPaths:
    hand: Path
    face: Path


class ModelStore:
    BASE_URL = "https://storage.googleapis.com/mediapipe-models"

    def __init__(self, directory: Path = PROJECT_ROOT / "assets" / "models") -> None:
        self.directory = directory

    def ensure(self) -> ModelPaths:
        self.directory.mkdir(parents=True, exist_ok=True)
        paths = ModelPaths(
            self.directory / "hand_landmarker.task",
            self.directory / "face_landmarker.task",
        )
        for name, path in (("hand", paths.hand), ("face", paths.face)):
            model = f"{name}_landmarker"
            url = f"{self.BASE_URL}/{model}/{model}/float16/1/{model}.task"
            self._download_if_missing(path, url)
        return paths

    @staticmethod
    def _download_if_missing(path: Path, url: str) -> None:
        if path.is_file() and path.stat().st_size > 0:
            return
        context = ssl.create_default_context(cafile=certifi.where())
        temporary_path = None
        try:
            with urllib.request.urlopen(url, context=context, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                    temporary_path = Path(output.name)
                    shutil.copyfileobj(response, output)
            temporary_path.replace(path)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                f"Failed to download model {path.name}. Check your internet connection "
                "and run: python -m pip install --upgrade certifi"
            ) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
