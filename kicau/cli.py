"""Command line options, kept independent of camera and model initialization."""

import argparse

from .config import AppConfig


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="Kicau Mania Dance Detector")
    parser.add_argument(
        "--debug", action="store_true",
        help="Show detection status, landmarks, and motion trajectory on the camera preview.",
    )
    args = parser.parse_args(argv)
    return AppConfig(debug=args.debug)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    # --help and argument validation should work without loading ML/GUI libraries.
    from .app import KicauApp

    KicauApp(config).run()
