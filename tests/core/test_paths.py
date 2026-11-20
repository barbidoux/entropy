"""Where ENTROPY finds its files (`entropy.core.paths`)."""

from pathlib import Path

import pytest

from entropy.core import paths


def test_content_dir_is_the_clone_content_folder() -> None:
    assert (paths.content_dir() / "settings.toml").is_file()


def test_missing_content_is_a_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A wheel install does not ship content/: the game must run from its clone."""
    monkeypatch.setattr(paths, "_REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="run ENTROPY from its git clone"):
        paths.content_dir()
