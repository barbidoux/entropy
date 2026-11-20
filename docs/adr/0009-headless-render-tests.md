# ADR 0009: Headless render tests on Mesa at OpenGL 4.5, shaders in GLSL 4.50

- Status: accepted
- Date: 2026-11-20

## Context

SPEC 9.2 puts rendering tests in every pull request, on CPU, "headless (software OpenGL,
small images)", and asked M0 to check that Mesa makes this feasible. The game itself
targets OpenGL 4.6 core (SPEC 7.1).

The first CI run (pull request #23, GitHub `ubuntu-latest`, Mesa llvmpipe under Xvfb)
could not create a 4.6 core context: GLX reported `GLXBadFBConfig`. Mesa's software
rasterizer, llvmpipe, offers OpenGL 4.5 core.

## Decision

- The game asks for OpenGL 4.6 core (`content/settings.toml`); a test checks it.
- Render tests take their window settings from the `window_settings` fixture, which asks
  for 4.6 unless `ENTROPY_TEST_GL_VERSION` says otherwise. CI sets it to 4.5 and runs
  the tests under Xvfb with Mesa llvmpipe. On the laptop they run at 4.6 on the NVIDIA
  driver.
- Shaders are written in GLSL 4.50 core, and the renderer uses no OpenGL 4.6 feature
  (SPIR-V shaders, `GL_ARB_gl_spirv`, polygon offset clamp, anisotropic filtering as
  core), so one code path runs on both.
- A feature that needs 4.6, or anything llvmpipe lacks, gets its own ADR saying how it is
  tested, and its render tests are marked `gpu` so they run on the laptop only.

## Alternatives considered

- **Skip render tests in CI:** they would then run only when the laptop is on (risk R13),
  and a broken shader could reach `main`.
- **Lower the game to OpenGL 4.5:** loses nothing today, but SPEC 7.1 names 4.6, and the
  laptop driver provides it; the game should not be limited by the test machine.
- **Zink (OpenGL on Vulkan) or a GPU runner:** Zink needs a Vulkan driver (lavapipe), more
  moving parts for the same 4.6 core; GPU runners cost money and do not match the laptop.

## Consequences

- Every pull request renders real images in CI.
- CI images come from a software rasterizer; tests compare colours and shapes with
  tolerances, never exact pixels between CI and the laptop.

## How we would know it is wrong

A shader or render pass passes in CI and fails on the laptop's NVIDIA driver (or the
reverse) more than once, or the renderer needs a 4.6-only feature for a core pass.
