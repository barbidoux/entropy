"""Start-up errors reach Matt (`entropy.app.main.main`): logged and shown in a message box."""

import logging
import sys
from pathlib import Path

import pytest

from entropy.app import main as app_main


def test_a_failed_start_is_logged_and_shown_in_french(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shown: list[tuple[str, str]] = []

    def failing_run(*args: object, **kwargs: object) -> None:
        raise ValueError("controls.forward: no key labelled 'Ñ' on this keyboard")

    monkeypatch.setattr(app_main, "run", failing_run)
    monkeypatch.setattr(app_main, "_show_error", lambda title, text: shown.append((title, text)))
    log_path = tmp_path / "entropy.log"

    assert app_main.main(log_path=log_path) == 1

    title, text = shown[0]
    assert "ENTROPY" in title
    assert "n'a pas pu démarrer" in text
    assert "controls.forward: no key labelled 'Ñ'" in text
    assert str(log_path) in text
    log = log_path.read_text(encoding="utf-8")
    assert "Traceback" in log
    assert "ValueError" in log


def test_a_normal_run_returns_zero_and_logs_to_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def quiet_run(*args: object, **kwargs: object) -> None:
        app_main._log.info("frame loop finished")

    monkeypatch.setattr(app_main, "run", quiet_run)
    log_path = tmp_path / "entropy.log"

    assert app_main.main(log_path=log_path) == 0
    assert "frame loop finished" in log_path.read_text(encoding="utf-8")


def test_main_gives_the_logging_set_up_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The log file is closed (Windows refuses to delete an open file) and the handlers go."""
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    monkeypatch.setattr(app_main, "run", lambda *args, **kwargs: None)
    log_path = tmp_path / "entropy.log"

    app_main.main(log_path=log_path)

    log_path.unlink()
    assert root.handlers == handlers_before
    assert root.level == level_before


@pytest.mark.skipif(sys.platform != "win32", reason="message boxes are Windows only")
def test_the_error_box_comes_to_the_front(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shortcut's console is minimised: the box must not open behind other windows."""
    calls: list[tuple[object, ...]] = []

    class FakeUser32:
        def MessageBoxW(self, *args: object) -> int:  # noqa: N802 (Win32 name)
            calls.append(args)
            return 1

    monkeypatch.setattr(app_main.ctypes, "WinDLL", lambda name: FakeUser32())

    app_main._show_error("ENTROPY", "boom")

    ((_, text, title, flags),) = calls
    assert (text, title) == ("boom", "ENTROPY")
    mb_iconerror, mb_setforeground, mb_topmost = 0x10, 0x10000, 0x40000
    assert isinstance(flags, int)
    assert flags & mb_iconerror and flags & mb_setforeground and flags & mb_topmost
