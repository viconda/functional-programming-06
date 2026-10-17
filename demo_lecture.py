"""Convenient entry point for all lecture 06 demonstrations."""

import runpy
import sys
from pathlib import Path


def main() -> None:
    """Run the implementation while keeping its local imports available."""
    directory = Path(__file__).resolve().parent / "lecture06"
    sys.path.insert(0, str(directory))
    runpy.run_path(str(directory / "demo_lecture.py"), run_name="__main__")


if __name__ == "__main__":
    main()
