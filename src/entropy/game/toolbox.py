"""The player's tools (SPEC 4.2, 4.5): which one is in hand, its power, its use.

Keys 1 to 9 choose a tool, the mouse wheel sets its power, the left button (with the
mouse captured) uses it. `apply` runs at step 1 of the frame (SPEC 5.9), with the
frame's simulated advance: it uses the tool in hand, then moves the water in flight
outside the liquid's grid (`Jet`) and hands the water that reached the tank to the
liquid.
"""

from __future__ import annotations

from typing import Any

from entropy.game.input import TOOL_ACTIONS, FrameInput
from entropy.game.tools.base import Aim, Inflow, Tool, ToolContext
from entropy.game.tools.jet import Domain, Jet
from entropy.game.tools.registry import discover_tools


class Toolbox:
    def __init__(
        self,
        tools: list[Tool],
        *,
        jet: Jet,
        domain: Domain | None,
        liquid: Any,
        particle_volume_m3: float,
    ) -> None:
        if not tools:
            raise ValueError("a toolbox needs a tool")
        self.tools = tools
        self.index = 0
        self.jet = jet
        self.domain = domain
        self.liquid = Inflow(liquid, particle_volume_m3) if liquid is not None else None
        self.particle_volume_m3 = particle_volume_m3
        self._held = False
        self._pressed = False
        self._released = False
        self._let_go: Tool | None = None  # a tool switched away from with the button held
        self._engaged: Tool | None = None  # the tool that got `on_press` and no release yet
        self._switch_press = False  # the new tool starts: chosen with the button held

    @classmethod
    def standard(
        cls,
        *,
        gravity_m_per_s2: tuple[float, float, float],
        domain: Domain | None,
        liquid: Any,
        particle_volume_m3: float,
    ) -> Toolbox:
        """Every tool with a content file, in the order of keys 1, 2, ... (the registry)."""
        return cls(
            discover_tools(),
            jet=Jet(gravity_m_per_s2),
            domain=domain,
            liquid=liquid,
            particle_volume_m3=particle_volume_m3,
        )

    @property
    def overflow_m3(self) -> float:
        """Water the tools could not add: the tank was full (`Inflow`)."""
        return self.liquid.lost_m3 if self.liquid is not None else 0.0

    @property
    def current(self) -> Tool:
        return self.tools[self.index]

    def handle(self, frame: FrameInput) -> None:
        """Choose and set the tool from this frame's input; remember its button.

        The rules (#74's and #77's reviews):
        - only a tool that was pressed ("engaged") is held and released;
        - choosing another tool releases the engaged one; the new one starts, as if
          clicked, if the button is still held and a tool was in use;
        - a release in the frame of a switch belongs to the old tool (the new one is not
          engaged, so it gets none)."""
        for number, action in enumerate(TOOL_ACTIONS):
            if action in frame.pressed and number < len(self.tools) and number != self.index:
                in_use = self._engaged is not None or self._let_go is not None
                if self._engaged is not None:
                    self._let_go = self._let_go or self._engaged
                self._switch_press = frame.tool_held and in_use
                self._pressed = False  # a click meant for the old tool
                self.index = number
        if frame.wheel_steps:
            self.current.change_power(frame.wheel_steps)
        if frame.alternate_pressed:
            self.current.on_alternate()
        self._pressed = self._pressed or frame.tool_pressed
        self._released = self._released or frame.tool_released
        if not frame.tool_held:
            self._switch_press = False
        self._held = frame.tool_held

    def apply(self, aim: Aim, dt_s: float) -> None:
        """Step 1 of the frame: use the tool for `dt_s` of simulated time, then land
        the water in flight."""
        ctx = ToolContext(
            aim=aim,
            liquid=self.liquid,
            jet=self.jet,
            particle_volume_m3=self.particle_volume_m3,
            domain=self.domain,
        )
        if self._let_go is not None:
            self._let_go.on_release(ctx)
            if self._engaged is self._let_go:
                self._engaged = None
            self._let_go = None
        tool = self.current
        pressed = self._pressed or self._switch_press
        if pressed and self._engaged is not tool:
            tool.on_press(ctx)
            self._engaged = tool
        if self._engaged is tool and (self._held or pressed):
            tool.on_hold(ctx, dt_s)
        if self._released and self._engaged is tool:
            tool.on_release(ctx)
            self._engaged = None
        self._pressed = self._released = self._switch_press = False
        if self.domain is not None and dt_s > 0.0:
            self.jet.advance(dt_s, self.domain, self.liquid)

    def forget_clicks(self) -> None:
        """While the world stands still (paused or frozen) no tool acts: a click then is
        dropped, not kept for when the world runs again, maybe with another tool. A tool
        already in use keeps its release, and a tool chosen while one was held still
        starts, when the world runs again."""
        self._pressed = False

    def palette(self) -> list[str]:
        """HUD lines: the tool in hand and its settings in SI units (SPEC 4.7)."""
        tool = self.current
        settings = ", ".join(
            f"{p.label} {tool.values[name]:g} {p.unit}" for name, p in tool.spec.parameters.items()
        )
        keys = "  ".join(f"{n + 1} {t.spec.label}" for n, t in enumerate(self.tools))
        power = tool.spec.parameters[tool.spec.power].label
        lines = [
            f"Outil : {tool.spec.label} ({settings})",
            f"{keys}   clic gauche : utiliser   clic droit : autre forme   molette : {power}",
        ]
        status = getattr(tool, "status", "")
        if status:
            lines.append(f"{tool.spec.label} : {status}")
        if self.overflow_m3 > 0.0:
            lines.append(f"Cuve pleine : {self.overflow_m3 * 1e3:,.0f} L perdus")
        return lines
