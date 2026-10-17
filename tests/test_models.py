import io
import ssl
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from kicau.models import ModelStore


class ModelStoreTests(unittest.TestCase):
    def test_cached_models_do_not_access_network(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hand_landmarker.task").write_bytes(b"hand model")
            (root / "face_landmarker.task").write_bytes(b"face model")
            with patch("kicau.models.urllib.request.urlopen") as request:
                paths = ModelStore(root).ensure()
            request.assert_not_called()
            self.assertEqual(paths.hand.read_bytes(), b"hand model")
            self.assertEqual(paths.face.read_bytes(), b"face model")

    def test_download_uses_verified_tls_and_replaces_empty_cache(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hand_landmarker.task"
            path.touch()
            with patch("kicau.models.urllib.request.urlopen", return_value=io.BytesIO(b"model")) as request:
                ModelStore._download_if_missing(path, "https://example.com/model")
            self.assertEqual(path.read_bytes(), b"model")
            context = request.call_args.kwargs["context"]
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)

    def test_interrupted_download_does_not_leave_partial_model(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hand_landmarker.task"
            response = MagicMock()
            response.__enter__.return_value = response
            response.read.side_effect = [b"partial", OSError("connection lost")]
            with patch("kicau.models.urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "Failed to download model"):
                    ModelStore._download_if_missing(path, "https://example.com/model")
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
