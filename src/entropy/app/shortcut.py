"""A Windows desktop shortcut that launches the game (SPEC 10.5): `uv run entropy-shortcut`.

The shortcut runs `uv run entropy` in the clone, minimised console, so the
game starts with one double-click. It is written with Windows Script Host
through PowerShell, which every Windows install has.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MINIMISED = 7


@dataclass(frozen=True)
class Shortcut:
    path: Path
    target: Path
    arguments: str
    working_dir: Path
    description: str
    window_style: int


def shortcut_for(repo_root: Path, *, desktop: Path, uv: Path) -> Shortcut:
    """The ENTROPY shortcut on `desktop`, running `uv` in `repo_root`."""
    return Shortcut(
        path=desktop / "ENTROPY.lnk",
        target=uv,
        arguments="run entropy",
        working_dir=repo_root,
        description="ENTROPY physics sandbox",
        window_style=_MINIMISED,
    )


def _quote(value: object) -> str:
    """A PowerShell single-quoted string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def powershell_script(shortcut: Shortcut) -> str:
    """PowerShell that creates (or replaces) the shortcut."""
    return "\n".join(
        [
            "$shell = New-Object -ComObject WScript.Shell",
            f"$link = $shell.CreateShortcut({_quote(shortcut.path)})",
            f"$link.TargetPath = {_quote(shortcut.target)}",
            f"$link.Arguments = {_quote(shortcut.arguments)}",
            f"$link.WorkingDirectory = {_quote(shortcut.working_dir)}",
            f"$link.Description = {_quote(shortcut.description)}",
            f"$link.WindowStyle = {shortcut.window_style}",
            "$link.Save()",
        ]
    )


def _desktop() -> Path:
    """The user's desktop folder, as Windows reports it (it may be redirected)."""
    output = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    return Path(output.strip())


def main() -> int:
    if sys.platform != "win32":
        print("The desktop shortcut is for Windows only.", file=sys.stderr)
        return 1
    uv = shutil.which("uv")
    if uv is None:
        print("uv is not on the PATH: install uv first.", file=sys.stderr)
        return 1
    shortcut = shortcut_for(_REPO_ROOT, desktop=_desktop(), uv=Path(uv))
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", powershell_script(shortcut)],
        check=True,
        timeout=30,
    )
    print(f"Shortcut created: {shortcut.path}")
    return 0
