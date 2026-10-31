import numpy as np
import pytest

from veronica.geometry.camera import Intrinsics
from veronica.pipeline.analysis import Analysis


def synthetic_room(W=320, H=240, f=200.0):
    """Ray-traced toy room with known geometry: camera 1.2 m above the floor looking
    level at a back wall 4 m away, and a 0.5×0.5×0.5 m box on the floor 2.5 m ahead.
    Returns an Analysis (depth + one box mask) like the neural stage would produce."""
    K = Intrinsics(f, f, W / 2, H / 2, W, H)
    rays = K.rays()  # z = 1
    cam_h = 1.2
    depth = np.full((H, W), np.inf, np.float32)
    # floor: y_cam = +cam_h (y down)
    with np.errstate(divide="ignore"):
        t_floor = np.where(rays[..., 1] > 1e-6, cam_h / rays[..., 1], np.inf)
    depth = np.minimum(depth, t_floor)
    depth = np.minimum(depth, np.full_like(depth, 4.0))  # back wall z = 4
    # box: x in [-0.25, 0.25], y in [cam_h-0.5, cam_h], z in [2.5, 3.0] — front face z=2.5, top y=cam_h-0.5
    box = np.zeros((H, W), bool)
    zf = 2.5
    x, y = rays[..., 0] * zf, rays[..., 1] * zf
    front = (np.abs(x) <= 0.25) & (y >= cam_h - 0.5) & (y <= cam_h)
    depth[front & (zf < depth)] = zf
    box |= front
    # top face y = cam_h - 0.5 for z in [2.5, 3.0]
    with np.errstate(divide="ignore", invalid="ignore"):
        zt = (cam_h - 0.5) / rays[..., 1]
    top = (rays[..., 1] > 0) & (zt >= 2.5) & (zt <= 3.0) & (np.abs(rays[..., 0] * zt) <= 0.25) & ~front
    depth[top] = np.minimum(depth[top], zt[top])
    box |= top
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = 120
    img[box] = (200, 60, 40)
    return Analysis(img, depth.astype(np.float32), f, f, W / 2, H / 2, "model", box[None], ["box"], [0.9], [0.95], ["detector"])


@pytest.fixture
def room():
    return synthetic_room()
