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
from entropy.game.tools.base import Aim, Tool, ToolContext, load_tool_spec
from entropy.game.tools.hose import Hose
from entropy.game.tools.jet import Domain, Jet


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
        self.liquid = liquid
        self.particle_volume_m3 = particle_volume_m3
        self._held = False
        self._pressed = False
        self._released = False

    @classmethod
    def standard(
        cls,
        *,
        gravity_m_per_s2: tuple[float, float, float],
        domain: Domain | None,
        liquid: Any,
        particle_volume_m3: float,
    ) -> Toolbox:
        """The tools of M1, in the order of keys 1, 2, ..."""
        return cls(
            [Hose(load_tool_spec("hose"))],
            jet=Jet(gravity_m_per_s2),
            domain=domain,
            liquid=liquid,
            particle_volume_m3=particle_volume_m3,
        )

    @property
    def current(self) -> Tool:
        return self.tools[self.index]

    def handle(self, frame: FrameInput) -> None:
        """Choose and set the tool from this frame's input; remember its button."""
        for number, action in enumerate(TOOL_ACTIONS):
            if action in frame.pressed and number < len(self.tools) and number != self.index:
                self._released = self._held  # the old tool lets go
                self.index = number
        if frame.wheel_steps:
            self.current.change_power(frame.wheel_steps)
        self._pressed = self._pressed or frame.tool_pressed
        self._released = self._released or frame.tool_released
        self._held = frame.tool_held

    def apply(self, aim: Aim, dt_s: float) -> None:
        """Step 1 of the frame: use the tool for `dt_s` of simulated time, then land
        the water in flight."""
        ctx = ToolContext(
            aim=aim, liquid=self.liquid, jet=self.jet, particle_volume_m3=self.particle_volume_m3
        )
        tool = self.current
        if self._pressed:
            tool.on_press(ctx)
        if self._held or self._pressed:
            tool.on_hold(ctx, dt_s)
        if self._released:
            tool.on_release(ctx)
        self._pressed = self._released = False
        if self.domain is not None and dt_s > 0.0:
            self.jet.advance(dt_s, self.domain, self.liquid)

    def palette(self) -> list[str]:
        """HUD lines: the tool in hand and its settings in SI units (SPEC 4.7)."""
        tool = self.current
        settings = ", ".join(
            f"{p.label} {tool.values[name]:g} {p.unit}" for name, p in tool.spec.parameters.items()
        )
        keys = "  ".join(f"{n + 1} {t.spec.label}" for n, t in enumerate(self.tools))
        power = tool.spec.parameters[tool.spec.power].label
        return [
            f"Outil : {tool.spec.label} ({settings})",
            f"{keys}   clic gauche : utiliser   molette : {power}",
        ]
