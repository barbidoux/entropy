"""Start-up errors reach Matt (`entropy.app.main.main`): logged and shown in a message box."""

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
