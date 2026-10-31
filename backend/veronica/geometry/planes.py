"""RANSAC plane extraction and gravity-aligned world frame estimation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    normal: np.ndarray  # unit, camera space
    d: float  # n·x + d = 0
    inliers: np.ndarray  # indices into the point array given to fit
    kind: str = "plane"  # floor | wall | ceiling | plane

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d


def fit_plane_lstsq(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, -float(n @ c)


def ransac_planes(
    pts: np.ndarray,
    max_planes: int = 4,
    min_inlier_frac: float = 0.06,
    thresh_rel: float = 0.012,
    iters: int = 400,
    seed: int = 0,
) -> list[Plane]:
    """Sequential RANSAC. The inlier threshold scales with depth (monocular depth error
    grows with distance), so it is expressed relative to each point's z."""
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(pts))
    planes: list[Plane] = []
    n_total = len(pts)
    tol = thresh_rel * np.maximum(pts[:, 2], 0.3)
    for _ in range(max_planes):
        if len(remaining) < max(500, min_inlier_frac * n_total):
            break
        P = pts[remaining]
        T = tol[remaining]
        best, best_n, best_d = None, None, None
        for _ in range(iters):
            s = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            d = -float(n @ s[0])
            inl = np.abs(P @ n + d) < T
            if best is None or inl.sum() > best.sum():
                best, best_n, best_d = inl, n, d
        if best is None or best.sum() < min_inlier_frac * n_total:
            break
        n, d = fit_plane_lstsq(P[best])
        inl = np.abs(P @ n + d) < T
        planes.append(Plane(n, d, remaining[inl]))
        remaining = remaining[~inl]
    return planes


@dataclass
class WorldFrame:
    """Rigid transform from OpenCV camera space to a Y-up world in meters."""
    R: np.ndarray  # 3x3, world_from_camera rotation
    t: np.ndarray  # 3, world_from_camera translation
    up_source: str  # "floor" | "walls" | "camera"

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.R.T + self.t

    def apply_dir(self, v: np.ndarray) -> np.ndarray:
        return v @ self.R.T

    def camera_matrix_gl(self) -> np.ndarray:
        """4x4 world_from_camera for an OpenGL camera (looks down -Z, +Y up)."""
        M = np.eye(4)
        M[:3, :3] = self.R @ np.diag([1.0, -1.0, -1.0])
        M[:3, 3] = self.t
        return M


def classify_planes(planes: list[Plane], pts: np.ndarray, obj_pts: np.ndarray | None = None) -> np.ndarray | None:
    """Label floor / ceiling / wall in place; return the up vector (camera space) or None.

    Without a floor (e.g. a close-up looking down at a table) the largest plane that the
    objects rest *on top of* becomes the ground reference and is labelled "support"."""
    cam_up = np.array([0.0, -1.0, 0.0])
    floor = None
    for p in planes:
        # orient normals towards the camera (origin is on the positive side)
        if p.d < 0:
            p.normal, p.d = -p.normal, -p.d
    cands = [p for p in planes if abs(p.normal @ cam_up) > np.cos(np.radians(40))]
    for p in sorted(cands, key=lambda p: -len(p.inliers)):
        c = pts[p.inliers].mean(0)
        if p.normal @ cam_up > 0 and c[1] > 0:  # normal points up, plane is below the camera
            floor = p
            break
    if floor is None and obj_pts is not None and len(obj_pts) > 50:
        for p in sorted(planes, key=lambda p: -len(p.inliers)):
            if p.normal @ cam_up < 0.15:
                continue  # must face roughly upward in the image to be something to stand on
            above = np.mean(obj_pts @ p.normal + p.d > 0.005)
            if above > 0.8:
                floor = p
                break
    up = floor.normal.copy() if floor is not None else None
    if up is None:
        verts = [p for p in planes if abs(p.normal @ cam_up) < np.sin(np.radians(25))]
        if len(verts) >= 2:
            u = np.cross(verts[0].normal, verts[1].normal)
            if np.linalg.norm(u) > 0.5:
                u /= np.linalg.norm(u)
                up = u if u @ cam_up > 0 else -u
    ref_up = up if up is not None else cam_up
    for p in planes:
        c = abs(p.normal @ ref_up)
        if p is floor:
            p.kind = "floor" if abs(p.normal @ cam_up) > np.cos(np.radians(40)) else "support"
        elif c > np.cos(np.radians(25)):
            p.kind = "ceiling" if (p.normal @ ref_up) < 0 else "plane"
        elif c < np.sin(np.radians(20)):
            p.kind = "wall"
    return up


def world_frame(up_cam: np.ndarray | None, floor: Plane | None, anchor_cam: np.ndarray, walls: list[Plane] | None = None) -> WorldFrame:
    """Build a right-handed Y-up frame with the origin on the floor beneath `anchor_cam`.

    Horizontal axes follow the room (Manhattan assumption): +Z is the normal of the largest
    wall, pointing back towards the camera, so translating along X slides along that wall.
    Without walls they follow the camera (X ≈ right, -Z ≈ forward)."""
    src = (floor.kind if floor.kind == "floor" else "support plane") if floor is not None else ("walls" if up_cam is not None else "camera")
    y = up_cam if up_cam is not None else np.array([0.0, -1.0, 0.0])
    y = y / np.linalg.norm(y)
    fwd = np.array([0.0, 0.0, 1.0])
    fwd_h = fwd - (fwd @ y) * y
    if np.linalg.norm(fwd_h) < 1e-3:  # camera looking straight down/up
        fwd_h = np.array([0.0, 1.0, 0.0]) - (np.array([0.0, 1.0, 0.0]) @ y) * y
    z = -fwd_h / np.linalg.norm(fwd_h)
    if walls:
        w = max(walls, key=lambda p: len(p.inliers))
        n_h = w.normal - (w.normal @ y) * y
        if np.linalg.norm(n_h) > 0.5:
            n_h /= np.linalg.norm(n_h)
            n_cam = n_h if n_h @ z > 0 else -n_h
            # prefer "looking at the wall" unless the wall is seen almost edge-on
            if n_cam @ z > np.cos(np.radians(65)):
                z = n_cam
            else:
                p = np.cross(y, n_h)
                z = p if p @ z > 0 else -p
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=0)  # rows: world axes expressed in camera space
    if floor is not None:
        # project anchor onto the floor plane
        origin_cam = anchor_cam - floor.distance(anchor_cam[None])[0] * floor.normal
    else:
        origin_cam = anchor_cam
    t = -R @ origin_cam
    return WorldFrame(R, t, src)
