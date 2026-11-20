"""Where ENTROPY finds its files.

The game runs from its git clone (`uv sync` installs the package in editable
mode), so content sits next to the source tree; a wheel would not ship it.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    """The git clone ENTROPY runs from."""
    return _REPO_ROOT


def out_dir() -> Path:
    """`out/`: logs and unofficial results, ignored by git."""
    return _REPO_ROOT / "out"


def content_dir() -> Path:
    """`content/`: materials, maps, settings (SPEC 10.2)."""
    content = _REPO_ROOT / "content"
    if not content.is_dir():
        raise FileNotFoundError(
            f"no content folder at {content}: run ENTROPY from its git clone (uv run entropy)"
        )
    return content
