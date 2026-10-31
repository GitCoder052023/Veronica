"""Triangle mesh container shared by the pipeline and the asset writer."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    positions: np.ndarray  # (V,3) float32
    indices: np.ndarray  # (F,3) uint32
    uvs: np.ndarray | None = None  # (V,2) float32, into the texture named by `texture`
    colors: np.ndarray | None = None  # (V,3) uint8
    observed: np.ndarray | None = None  # (V,) uint8: 255 observed, 0 inferred
    texture: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return len(self.positions)

    @property
    def n_faces(self) -> int:
        return len(self.indices)

    def compact(self) -> "Mesh":
        """Drop unreferenced vertices."""
        if self.n_faces == 0:
            return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), texture=self.texture)
        used = np.unique(self.indices)
        remap = np.full(self.n_vertices, -1, np.int64)
        remap[used] = np.arange(len(used))
        sel = lambda a: None if a is None else a[used]
        return Mesh(
            self.positions[used].astype(np.float32),
            remap[self.indices].astype(np.uint32),
            sel(self.uvs), sel(self.colors), sel(self.observed), self.texture, dict(self.extra),
        )

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "Mesh":
        m = Mesh(**{**self.__dict__})
        m.positions = (self.positions @ R.T + t).astype(np.float32)
        if np.linalg.det(R) < 0:
            m.indices = self.indices[:, ::-1].copy()
        return m

    def face_normals(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit face normals and face areas."""
        p = self.positions[self.indices]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        a = np.linalg.norm(n, axis=1)
        return n / np.maximum(a, 1e-12)[:, None], 0.5 * a

    @staticmethod
    def concat(meshes: list["Mesh"]) -> "Mesh":
        meshes = [m for m in meshes if m.n_faces]
        if not meshes:
            return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))
        off = np.cumsum([0] + [m.n_vertices for m in meshes[:-1]])
        cat = lambda k: None if any(getattr(m, k) is None for m in meshes) else np.concatenate([getattr(m, k) for m in meshes])
        return Mesh(
            np.concatenate([m.positions for m in meshes]).astype(np.float32),
            np.concatenate([m.indices + o for m, o in zip(meshes, off)]).astype(np.uint32),
            cat("uvs"), cat("colors"), cat("observed"), meshes[0].texture,
        )


def grid_mesh(
    points: np.ndarray,
    valid: np.ndarray,
    stride: int = 1,
    max_edge_ratio: float = 0.05,
    observed: np.ndarray | None = None,
    texture: str | None = None,
) -> Mesh:
    """Triangulate an organised point map (H,W,3) on its pixel grid.

    Two triangles per valid quad; a triangle is rejected when its depth ratio exceeds
    1 + max_edge_ratio·stride, i.e. it would bridge an occlusion boundary.
    UVs point back into the source image so the mesh carries full-resolution texture.
    """
    H, W = valid.shape
    P = points[::stride, ::stride]
    V = valid[::stride, ::stride]
    h, w = V.shape
    idx = np.arange(h * w).reshape(h, w)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    Vf = V.ravel()
    ok = Vf[tris].all(1)
    z = P[..., 2].ravel()
    zt = z[tris]
    ok &= zt.max(1) / np.maximum(zt.min(1), 1e-6) < 1.0 + max_edge_ratio * stride
    tris = tris[ok]
    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    uv = np.stack([(us.ravel() + 0.5) / W, (vs.ravel() + 0.5) / H], 1).astype(np.float32)
    obs = None
    if observed is not None:
        obs = np.where(observed[::stride, ::stride].ravel(), 255, 0).astype(np.uint8)
    m = Mesh(P.reshape(-1, 3).astype(np.float32), tris.astype(np.uint32), uv, None, obs, texture)
    return m.compact()


def auto_stride(n_pixels: int, budget: int) -> int:
    s = 1
    while n_pixels / (s * s) > budget:
        s += 1
    return s
