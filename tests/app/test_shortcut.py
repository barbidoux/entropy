"""Desktop shortcut to launch the game (`entropy.app.shortcut`), SPEC 10.5."""

import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from entropy.app import shortcut
from entropy.app.shortcut import Shortcut, powershell_script, shortcut_for

REPO = Path(r"C:\Users\matth\GITREPOS\entropy")
DESKTOP = Path(r"C:\Users\matth\Desktop")
UV = Path(r"C:\Users\matth\.local\bin\uv.exe")


def test_shortcut_runs_the_game_with_uv_from_the_clone() -> None:
    shortcut = shortcut_for(REPO, desktop=DESKTOP, uv=UV)

    assert shortcut == Shortcut(
        path=PureWindowsPath(str(DESKTOP)) / "ENTROPY.lnk",
        target=PureWindowsPath(str(UV)),
        arguments="run entropy",
        working_dir=PureWindowsPath(str(REPO)),
        description="ENTROPY physics sandbox",
        window_style=7,
    )


def test_script_sets_every_shortcut_field() -> None:
    script = powershell_script(shortcut_for(REPO, desktop=DESKTOP, uv=UV))

    assert "WScript.Shell" in script
    assert rf"'{DESKTOP}\ENTROPY.lnk'" in script
    assert f".TargetPath = '{UV}'" in script
    assert ".Arguments = 'run entropy'" in script
    assert f".WorkingDirectory = '{REPO}'" in script
    assert ".WindowStyle = 7" in script
    assert script.rstrip().endswith(".Save()")


def test_main_runs_the_script_for_the_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=str(DESKTOP) + "\n")

    monkeypatch.setattr(shortcut.sys, "platform", "win32")
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(shortcut.shutil, "which", lambda name: str(UV))
    monkeypatch.setattr(shortcut.subprocess, "run", fake_run)

    assert shortcut.main() == 0
    script = calls[-1][-1]
    assert rf"'{DESKTOP}\ENTROPY.lnk'" in script
    assert f".TargetPath = '{UV}'" in script
    assert f".WorkingDirectory = '{PureWindowsPath(str(shortcut._REPO_ROOT))}'" in script


def test_main_prefers_the_uv_that_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=str(DESKTOP) + "\n")

    monkeypatch.setattr(shortcut.sys, "platform", "win32")
    monkeypatch.setenv("UV", r"D:\tools\uv.exe")
    monkeypatch.setattr(shortcut.subprocess, "run", fake_run)

    assert shortcut.main() == 0
    assert r".TargetPath = 'D:\tools\uv.exe'" in calls[-1][-1]


def test_main_explains_what_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(shortcut.sys, "platform", "linux")
    assert shortcut.main() == 1
    assert "Windows only" in capsys.readouterr().err
    monkeypatch.setattr(shortcut.sys, "platform", "win32")
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(shortcut.shutil, "which", lambda name: None)
    assert shortcut.main() == 1

    assert "uv is not on the PATH" in capsys.readouterr().err


def test_windows_separators_whatever_system_builds_the_script() -> None:
    """CI runs on Linux, where Path joins with '/'."""
    shortcut = shortcut_for(
        PurePosixPath(str(REPO)), desktop=PurePosixPath(str(DESKTOP)), uv=PurePosixPath(str(UV))
    )

    assert str(shortcut.path) == rf"{DESKTOP}\ENTROPY.lnk"


def test_quotes_in_paths_are_escaped_for_powershell() -> None:
    shortcut = shortcut_for(Path(r"C:\Users\o'neil\entropy"), desktop=DESKTOP, uv=UV)

    assert r"'C:\Users\o''neil\entropy'" in powershell_script(shortcut)
