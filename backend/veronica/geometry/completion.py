"""Amodal completion of the unseen side of an object from a single view.

A single photograph observes only the camera-facing surface. These completers add
an *inferred* back side using explicit, inspectable shape priors, and mark every
vertex they invent with observed=0 so the UI can render it differently:

* RevolveCompleter — rotationally symmetric objects (bottles, cups, vases): fits a
  vertical axis to the visible arc and sweeps the observed radius profile.
* MirrorCompleter — everything else: reflects the observed surface across the
  object's front/back and left/right vertical symmetry planes (gravity aligned),
  with the depth extent taken from visible side faces or a per-class prior.

Neither invents detail that is not in the photo; a learned generative completer
(TRELLIS, SAM 3D Objects, Hunyuan3D) can replace them behind the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh

# Depth/width ratio priors used only when no side face of the object is visible.
DEPTH_PRIORS: dict[str, float] = {
    "table": 0.6, "desk": 0.55, "chair": 1.0, "sofa": 0.45, "bed": 1.3, "cabinet": 0.5, "shelf": 0.35,
    "bookshelf": 0.3, "drawer": 0.9, "door": 0.04, "window": 0.05, "curtain": 0.05, "computer monitor": 0.25,
    "television": 0.12, "laptop": 0.7, "keyboard": 0.35, "computer mouse": 1.7, "computer tower": 2.0,
    "speaker": 1.0, "phone": 0.1, "tablet": 0.08, "book": 0.7, "notebook": 0.7, "box": 1.0, "bag": 0.5,
    "backpack": 0.5, "pillow": 0.35, "clock": 0.15, "picture frame": 0.04, "poster": 0.01, "mirror": 0.04,
    "power socket": 0.3, "light switch": 0.3, "remote control": 0.3, "printer": 0.8, "router": 0.8,
    "refrigerator": 1.0, "microwave": 0.7, "oven": 1.0, "rug": 1.0, "stool": 1.0, "bench": 0.35,
}
REVOLVE_CLASSES = {"bottle", "cup", "mug", "glass", "vase", "bowl", "plate", "saucer", "can", "jar", "candle", "trash can", "kettle", "pot", "potted plant", "lamp"}
FLAT_CLASSES = {"door", "window", "poster", "picture frame", "mirror", "power socket", "light switch", "curtain", "rug"}


@dataclass
class ObjectContext:
    label: str
    surface: Mesh  # observed surface in world space (Y up, meters)
    camera_pos: np.ndarray  # world-space camera center
    floor_y: float | None
    point_spacing: float  # typical distance between neighbouring observed points (m)


@dataclass
class Completion:
    mesh: Mesh | None
    method: str
    params: dict


class Completer(Protocol):
    name: str

    def complete(self, ctx: ObjectContext) -> Completion: ...


def _yaw_frame(ctx: ObjectContext) -> tuple[np.ndarray, np.ndarray, float]:
    """Return horizontal unit axes (a, b): `a` is the object's front→back axis, `b` its
    left→right axis, plus the orientation confidence in [0,1]."""
    n, area = ctx.surface.face_normals()
    h = n[:, [0, 2]]
    hn = np.linalg.norm(h, axis=1)
    w = area * hn  # weight vertical faces (horizontal normals) by area
    theta = np.arctan2(h[:, 1], h[:, 0])
    c, s = (w * np.cos(4 * theta)).sum(), (w * np.sin(4 * theta)).sum()
    conf = float(np.hypot(c, s) / max(w.sum(), 1e-9))
    center = ctx.surface.positions.mean(0)
    view = center - ctx.camera_pos
    view_h = np.array([view[0], view[2]])
    view_h /= max(np.linalg.norm(view_h), 1e-9)
    if conf > 0.25:
        t0 = np.arctan2(s, c) / 4.0
        # Of the two Manhattan axes, the front/back one is the normal of the largest visible
        # vertical face (e.g. a monitor's screen even when seen at 40°), not the axis that
        # happens to be closest to the viewing direction.
        hu = h / np.maximum(hn, 1e-9)[:, None]
        scores = []
        for k in range(2):
            ax = np.array([np.cos(t0 + k * np.pi / 2), np.sin(t0 + k * np.pi / 2)])
            scores.append((w * (np.abs(hu @ ax) > np.cos(np.pi / 4))).sum())
        k = int(np.argmax(scores))
        a2 = np.array([np.cos(t0 + k * np.pi / 2), np.sin(t0 + k * np.pi / 2)])
        if a2 @ view_h < 0:  # orient front → back (away from the camera)
            a2 = -a2
    else:
        a2 = view_h
    a = np.array([a2[0], 0.0, a2[1]])
    b = np.array([-a2[1], 0.0, a2[0]])
    return a, b, conf


def _reflect(mesh: Mesh, normal: np.ndarray, offset: float) -> Mesh:
    """Reflect across the plane {x : x·normal = offset}; flip winding to keep outward faces."""
    p = mesh.positions
    d = p @ normal - offset
    m = Mesh(**{**mesh.__dict__})
    m.positions = (p - 2.0 * d[:, None] * normal[None]).astype(np.float32)
    m.indices = mesh.indices[:, ::-1].copy()
    m.observed = np.zeros(mesh.n_vertices, np.uint8)
    return m


def _drop_overlapping(cand: Mesh, ref_pts: np.ndarray, radius: float) -> Mesh:
    """Remove candidate triangles that lie on top of geometry we already have."""
    if cand.n_faces == 0 or len(ref_pts) == 0:
        return cand
    dist, _ = cKDTree(ref_pts).query(cand.positions, k=1, distance_upper_bound=radius * 4)
    near = dist < radius
    keep = ~near[cand.indices].all(1)
    m = Mesh(**{**cand.__dict__})
    m.indices = cand.indices[keep]
    return m.compact()


class MirrorCompleter:
    name = "mirror-symmetry"

    def complete(self, ctx: ObjectContext) -> Completion:
        S = ctx.surface
        if S.n_faces < 8:
            return Completion(None, "none", {"reason": "too little observed surface"})
        a, b, conf = _yaw_frame(ctx)
        P = S.positions
        pa, pb = P @ a, P @ b
        a_lo, a_hi = np.percentile(pa, [2, 98])
        b_lo, b_hi = np.percentile(pb, [2, 98])
        width = float(b_hi - b_lo)
        prior = DEPTH_PRIORS.get(ctx.label, 0.8) * width
        observed_depth = float(a_hi - a_lo)
        side_visible = observed_depth > 0.5 * prior
        depth = observed_depth if side_visible else max(prior, observed_depth)
        a_mid = float(a_lo + depth / 2.0)
        b_mid = float((b_lo + b_hi) / 2.0)

        radius = max(1.5 * ctx.point_spacing, 0.004)
        parts: list[Mesh] = []
        ref = P.copy()
        for refl in (lambda m: _reflect(m, a, a_mid), lambda m: _reflect(m, b, b_mid), lambda m: _reflect(_reflect(m, a, a_mid), b, b_mid)):
            cand = _drop_overlapping(refl(S), ref, radius)
            if ctx.floor_y is not None and cand.n_faces:
                below = cand.positions[:, 1] < ctx.floor_y - 0.01
                keep = ~below[cand.indices].any(1)
                cand.indices = cand.indices[keep]
                cand = cand.compact()
            if cand.n_faces:
                parts.append(cand)
                ref = np.concatenate([ref, cand.positions])
        mesh = Mesh.concat(parts) if parts else None
        if mesh is not None:
            mesh.texture = S.texture
        return Completion(
            mesh, self.name,
            {
                "front_axis": a.round(4).tolist(), "side_axis": b.round(4).tolist(), "orientation_confidence": round(conf, 3),
                "depth_m": round(depth, 4), "depth_source": "observed side faces" if side_visible else f"class prior ({ctx.label})",
                "width_m": round(width, 4),
            },
        )


def _fit_circle(xz: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic (Kåsa) circle fit; returns centre, radius and RMS residual."""
    A = np.c_[2 * xz, np.ones(len(xz))]
    y = (xz**2).sum(1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    c = sol[:2]
    r = float(np.sqrt(max(sol[2] + c @ c, 1e-12)))
    res = float(np.sqrt(np.mean((np.linalg.norm(xz - c, axis=1) - r) ** 2)))
    return c, r, res


class RevolveCompleter:
    name = "surface-of-revolution"
    segments = 48
    rings = 40

    def complete(self, ctx: ObjectContext) -> Completion:
        S = ctx.surface
        P = S.positions
        if len(P) < 50:
            return Completion(None, "none", {"reason": "too few points"})
        y = P[:, 1]
        y0, y1 = np.percentile(y, [1, 99])
        if ctx.floor_y is not None and y0 - ctx.floor_y < 0.03:
            y0 = ctx.floor_y
        c, r, res = _fit_circle(P[:, [0, 2]])
        width = np.ptp(P[:, [0, 2]] @ np.array([[1, 0], [0, 1]]), axis=0).max()
        cam_h = np.array([ctx.camera_pos[0], ctx.camera_pos[2]])
        # A believable axis lies behind the visible arc, and the radius matches its width.
        # The axis must not sit in front of the visible surface (side views see an arc whose
        # centre is behind it; top-down views see a full disc whose centre is at it).
        behind = np.linalg.norm(c - cam_h) > np.linalg.norm(P[:, [0, 2]].mean(0) - cam_h) - 0.25 * r
        if not (behind and 0.25 * width < r < 1.2 * width and res < 0.35 * r):
            return MirrorCompleter().complete(ctx)

        edges = np.linspace(y0, y1, self.rings + 1)
        rad = np.linalg.norm(P[:, [0, 2]] - c, axis=1)
        prof = np.full(self.rings, np.nan)
        for i in range(self.rings):
            sel = (y >= edges[i]) & (y < edges[i + 1])
            if sel.sum() >= 3:
                prof[i] = np.percentile(rad[sel], 60)  # robust to handles / labels sticking out
        if np.isnan(prof).all():
            return MirrorCompleter().complete(ctx)
        idx = np.arange(self.rings)
        ok = ~np.isnan(prof)
        prof = np.interp(idx, idx[ok], prof[ok])
        prof = np.convolve(np.pad(prof, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        ys = (edges[:-1] + edges[1:]) / 2

        ang = np.linspace(0, 2 * np.pi, self.segments, endpoint=False)
        ring_pts = np.stack(
            [c[0] + prof[:, None] * np.cos(ang)[None], np.repeat(ys[:, None], self.segments, 1), c[1] + prof[:, None] * np.sin(ang)[None]], -1
        ).reshape(-1, 3)
        R, Sg = self.rings, self.segments
        g = np.arange(R * Sg).reshape(R, Sg)
        a, b = g[:-1], np.roll(g[:-1], -1, axis=1)
        cc, d = g[1:], np.roll(g[1:], -1, axis=1)
        tris = np.concatenate([np.stack([a, cc, b], -1).reshape(-1, 3), np.stack([b, cc, d], -1).reshape(-1, 3)])
        # caps
        top_c, bot_c = len(ring_pts), len(ring_pts) + 1
        ring_pts = np.vstack([ring_pts, [c[0], ys[-1], c[1]], [c[0], ys[0], c[1]]])
        top = np.stack([np.full(Sg, top_c), g[-1], np.roll(g[-1], -1)], 1)
        bot = np.stack([np.full(Sg, bot_c), np.roll(g[0], -1), g[0]], 1)
        tris = np.concatenate([tris, top, bot]).astype(np.uint32)

        # Texture: vertices near observed points borrow that point's UV; hidden ones borrow the
        # UV of their mirror image across the plane containing the axis and the camera.
        tree = cKDTree(P)
        uv_src = S.uvs if S.uvs is not None else np.zeros((len(P), 2), np.float32)
        dist, nn = tree.query(ring_pts, k=1)
        observed = dist < max(3 * ctx.point_spacing, 0.15 * float(prof.mean()))
        view = np.array([c[0] - cam_h[0], 0.0, c[1] - cam_h[1]])
        view /= np.linalg.norm(view)
        axis_c = np.array([c[0], 0.0, c[1]])
        rel = ring_pts - axis_c
        mirrored = ring_pts - 2.0 * (rel @ view)[:, None] * view[None]
        _, nn_m = tree.query(mirrored, k=1)
        uvs = np.where(observed[:, None], uv_src[nn], uv_src[nn_m]).astype(np.float32) if S.uvs is not None else None
        cols = np.where(observed[:, None], S.colors[nn], S.colors[nn_m]).astype(np.uint8) if S.colors is not None else None
        mesh = Mesh(ring_pts.astype(np.float32), tris, uvs, cols, np.where(observed, 255, 0).astype(np.uint8), S.texture)
        return Completion(
            mesh, self.name,
            {"axis_xz": c.round(4).tolist(), "mean_radius_m": round(float(prof.mean()), 4), "fit_rms_m": round(res, 4), "height_m": round(float(y1 - y0), 4)},
        )


def completer_for(label: str) -> Completer:
    return RevolveCompleter() if label in REVOLVE_CLASSES else MirrorCompleter()
