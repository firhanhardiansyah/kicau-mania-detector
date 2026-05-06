import pygame


class AudioPlayer:
    def __init__(self, music_path: str):
        self.music_path = music_path
        self._initialized = False

    def _init_mixer(self) -> None:
        if self._initialized:
            return
        pygame.mixer.init()
        pygame.mixer.music.load(self.music_path)
        self._initialized = True

    def play_once(self) -> None:
        self._init_mixer()
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(0)

    def is_playing(self) -> bool:
        if not self._initialized:
            return False
        return pygame.mixer.music.get_busy()

    def stop(self) -> None:
        if not self._initialized:
            return
        pygame.mixer.music.stop()

    def shutdown(self) -> None:
        if not self._initialized:
            return
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        self._initialized = False
