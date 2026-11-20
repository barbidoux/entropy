"""Where ENTROPY finds its files."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def content_dir() -> Path:
    """`content/`: materials, maps, settings (SPEC 10.2). The game runs from its clone."""
    return _REPO_ROOT / "content"
