"""View and projection matrices for a camera in the Z-up world frame (ADR 0003).

Matrices are float32 NumPy arrays in the usual mathematical layout
(`clip = projection @ view @ point`); `gl_bytes` converts them to the
column-major layout OpenGL expects. Clip space follows OpenGL: x right,
y up, depth from -1 (near plane) to +1 (far plane).
"""

import numpy as np
import numpy.typing as npt

WORLD_UP = (0.0, 0.0, 1.0)

type Vector = npt.ArrayLike


def view_matrix(eye_m: Vector, forward: Vector, up: Vector = WORLD_UP) -> np.ndarray:
    """World to camera space for a camera at `eye_m` (m) looking along `forward`.

    `up` is the world direction that appears upwards on screen; `forward` must
    not be parallel to it.
    """
    f = np.asarray(forward, dtype=np.float64)
    f = f / np.linalg.norm(f)
    side = np.cross(f, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(side) < 1e-9:
        raise ValueError("the view direction is parallel to the up direction")
    side /= np.linalg.norm(side)
    true_up = np.cross(side, f)
    view = np.identity(4)
    view[0, :3], view[1, :3], view[2, :3] = side, true_up, -f
    view[:3, 3] = -view[:3, :3] @ np.asarray(eye_m, dtype=np.float64)
    return view.astype(np.float32)


def perspective(fov_y_rad: float, aspect: float, near_m: float, far_m: float) -> np.ndarray:
    """OpenGL perspective projection; `fov_y_rad` is the vertical field of view."""
    if not (0.0 < fov_y_rad < np.pi and aspect > 0.0 and 0.0 < near_m < far_m):
        raise ValueError(
            f"no projection for fov {fov_y_rad} rad, aspect {aspect}, "
            f"near {near_m} m, far {far_m} m"
        )
    f = 1.0 / np.tan(fov_y_rad / 2.0)
    projection = np.zeros((4, 4))
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (far_m + near_m) / (near_m - far_m)
    projection[2, 3] = 2.0 * far_m * near_m / (near_m - far_m)
    projection[3, 2] = -1.0
    return projection.astype(np.float32)


def gl_bytes(matrix: np.ndarray) -> bytes:
    """A 4x4 matrix as the column-major float32 bytes of a GLSL `mat4` uniform."""
    return np.ascontiguousarray(matrix.T, dtype=np.float32).tobytes()
