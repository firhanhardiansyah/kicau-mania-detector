import unittest
from unittest.mock import patch

import pygame

from kicau.audio import AudioPlayer


class AudioTests(unittest.TestCase):
    def test_invalid_music_closes_mixer_and_allows_retry(self):
        with patch("kicau.audio.pygame.mixer") as mixer:
            player = AudioPlayer("music.mp3")
            mixer.music.load.side_effect = pygame.error("invalid audio")
            with self.assertRaisesRegex(pygame.error, "invalid audio"):
                player.play_once()
            mixer.quit.assert_called_once()
            mixer.music.load.side_effect = None
            mixer.music.get_busy.return_value = False
            player.play_once()
            self.assertEqual(mixer.init.call_count, 2)
            mixer.music.play.assert_called_once_with(0)

    def test_shutdown_before_initialization_is_safe(self):
        with patch("kicau.audio.pygame.mixer") as mixer:
            AudioPlayer("music.mp3").shutdown()
            mixer.quit.assert_not_called()
