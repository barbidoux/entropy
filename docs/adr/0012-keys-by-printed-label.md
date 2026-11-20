# ADR 0012: Keys are set by the labels printed on the keyboard

- Status: accepted
- Date: 2026-11-20

## Context

SPEC 4.5 names the controls as on a QWERTY keyboard (W A S D). Matt's keyboard is AZERTY
and he asked for a way to set the keys (issue #32). GLFW reports keys as physical
positions named after the US layout, so the default W A S D tokens already sit where
Z Q S D are printed on his keyboard; but a player who writes "Z" in a settings file
means the key marked Z.

## Decision

- Keys are set in the `[controls]` table of `content/settings.toml`, one entry per action.
- A single letter means the key printed with that letter in the current keyboard layout,
  found with `glfw.get_key_name`. A digit means the key printed with it, or, when no key
  prints it unshifted (AZERTY), the digit row. `@W` names a position: the key where W
  sits on a US keyboard. Other keys use GLFW's names (SPACE, LEFT_CONTROL, F3...).
  Several keys for one action are separated by commas.
- The code defaults are positions (`@W @A @S @D`), so they give W A S D on QWERTY and
  Z Q S D on AZERTY. The shipped settings write Matt's keys as he sees them: Z Q S D.
- An unknown name, a missing label or a key bound to two actions stops the game with a
  message box naming the setting (the console of the desktop shortcut is minimised).

## Alternatives considered

- **Positions only (GLFW tokens):** works on every layout without settings, but a player
  writing "Z" would get the key where Z sits on a US keyboard, which on AZERTY is W.
- **Labels only:** the defaults would then depend on the layout and give the wrong
  positions on AZERTY.

## Consequences

- Settings read the way the player sees the keyboard; defaults work on any layout.
- Label lookups need GLFW running, so bindings are built after the window opens.

## How we would know it is wrong

Players on other layouts (QWERTZ, Dvorak) find keys that cannot be named, or a layout
change while the game runs leaves the bindings on the wrong keys.
