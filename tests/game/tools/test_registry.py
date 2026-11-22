"""Tools found through a registry (SPEC 8.2, #53): content files and registered classes."""

from pathlib import Path

import pytest

from entropy.game.tools.base import ToolBase, ToolSpec
from entropy.game.tools.registry import discover_tools, register_tool


@register_tool("test_tap")
class _Tap(ToolBase):
    """A tool only these tests know."""

    def __init__(self, spec: ToolSpec) -> None:
        super().__init__(spec)


def _write_tool(folder: Path, name: str, order: int) -> None:
    (folder / f"{name}.toml").write_text(
        f"""name = "{name}"
label = "{name}"
icon = "t"
power = "rate"
order = {order}

[parameters.rate]
label = "rate"
unit = "1/s"
minimum = 0.0
maximum = 1.0
default = 0.5
step = 0.1
""",
        encoding="utf-8",
    )


def test_the_m1_tools_come_from_their_content_files_in_key_order() -> None:
    names = [tool.spec.name for tool in discover_tools()]

    assert names == ["hose", "bucket", "flood", "eraser"]


def test_a_new_tool_needs_only_its_class_and_its_content_file(tmp_path: Path) -> None:
    """Adding a tool names it nowhere else: the toolbox never names a specific tool."""
    _write_tool(tmp_path, "test_tap", order=1)

    tools = discover_tools(tmp_path)

    assert [type(tool) for tool in tools] == [_Tap]
    assert tools[0].values == {"rate": 0.5}


def test_a_content_file_without_a_registered_class_is_refused(tmp_path: Path) -> None:
    _write_tool(tmp_path, "nobody_made_me", order=1)

    with pytest.raises(ValueError, match="nobody_made_me"):
        discover_tools(tmp_path)


def test_two_tools_on_one_key_are_refused(tmp_path: Path) -> None:
    _write_tool(tmp_path, "test_tap", order=1)
    _write_tool(tmp_path, "hose", order=1)

    with pytest.raises(ValueError, match="order"):
        discover_tools(tmp_path)


@pytest.mark.parametrize("order", [0, 10])
def test_a_tool_s_key_is_1_to_9(tmp_path: Path, order: int) -> None:
    _write_tool(tmp_path, "test_tap", order=order)

    with pytest.raises(ValueError, match="1 to 9"):
        discover_tools(tmp_path)
