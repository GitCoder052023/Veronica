"""Pinhole camera helpers. Camera space follows OpenCV: +x right, +y down, +z forward."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def rays(self) -> np.ndarray:
        """(H, W, 3) un-normalised ray directions with z = 1."""
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32))
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=-1)

    def unproject(self, depth: np.ndarray) -> np.ndarray:
        return self.rays() * depth[..., None]

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Camera-space points (N,3) → pixel u, v and depth z."""
        z = pts[:, 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        return pts[:, 0] / zs * self.fx + self.cx, pts[:, 1] / zs * self.fy + self.cy, z

    def to_dict(self) -> dict:
        return dict(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, width=self.width, height=self.height)


def depth_discontinuities(depth: np.ndarray, rel_thresh: float = 0.035) -> np.ndarray:
    """Pixels whose 4-neighbourhood jumps by more than rel_thresh·z — occlusion edges
    where monocular depth produces 'flying pixels' between foreground and background."""
    z = np.maximum(depth, 1e-4)
    edge = np.zeros(z.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = z[: z.shape[0] - dy, : z.shape[1] - dx]
        b = z[dy:, dx:]
        jump = np.abs(a - b) / np.minimum(a, b) > rel_thresh
        edge[: z.shape[0] - dy, : z.shape[1] - dx] |= jump
        edge[dy:, dx:] |= jump
    return edge
