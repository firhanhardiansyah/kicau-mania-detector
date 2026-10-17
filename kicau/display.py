"""Fullscreen output and keyboard events. OpenCV is used only for image conversion."""

from enum import Enum, auto

import cv2
import numpy as np
import pygame


class Command(Enum):
    QUIT = auto()
    PREVIEW = auto()


class PreviewWindow:
    def __init__(self, title: str) -> None:
        pygame.display.init()
        try:
            pygame.display.set_caption(title)
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        except Exception:
            pygame.display.quit()
            raise

    @property
    def size(self) -> tuple[int, int]:
        return self.screen.get_size()

    def show(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame.surfarray.blit_array(self.screen, rgb.swapaxes(0, 1))
        pygame.display.flip()

    def commands(self) -> list[Command]:
        commands = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
            ):
                commands.append(Command.QUIT)
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                commands.append(Command.PREVIEW)
        return commands

    def shutdown(self) -> None:
        pygame.display.quit()
