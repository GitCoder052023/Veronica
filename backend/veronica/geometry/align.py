"""Place a generated object (canonical frame, arbitrary scale/yaw) onto its observed
metric surface in the reconstructed scene.

Image-to-3D generators (TRELLIS, Hunyuan3D, …) return one object, upright (+Y) but
with unknown size, heading and position. We know three things about the real object:

* its observed surface points (metric, gravity-aligned world frame),
* its silhouette in every photo it was segmented in,
* what it rests on (floor / support plane / another object).

Alignment = search heading over 360°, and for each heading fit scale + translation by
trimmed ICP from the observed points to the generated surface; score each hypothesis by
surface distance *and* silhouette agreement (the silhouette is what disambiguates
front from back), then refine the best one with a yaw+scale+translation Procrustes ICP.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .camera import Intrinsics


@dataclass
class ViewSilhouette:
    K: Intrinsics
    R_cw: np.ndarray  # camera_from_world rotation
    t_cw: np.ndarray
    mask: np.ndarray  # (H,W) bool object mask in this view


@dataclass
class Alignment:
    matrix: np.ndarray  # 4x4 world_from_canonical (similarity)
    yaw_deg: float
    scale: float
    surface_rms_m: float  # trimmed RMS distance observed → generated surface
    silhouette_iou: float  # mean over views
    score: float
    accepted: bool
    reason: str = ""
    candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "yaw_deg": round(self.yaw_deg, 2), "scale": round(self.scale, 5), "surface_rms_m": round(self.surface_rms_m, 4),
            "silhouette_iou": round(self.silhouette_iou, 3), "score": round(self.score, 4), "accepted": self.accepted, "reason": self.reason,
        }


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _silhouette_iou(P: np.ndarray, views: list[ViewSilhouette], splat: int = 2) -> float:
    if not views:
        return 1.0
    ious = []
    for v in views:
        pc = P @ v.R_cw.T + v.t_cw
        u, w, z = v.K.project(pc)
        ok = (z > 0.05) & (u >= 0) & (u < v.K.width) & (w >= 0) & (w < v.K.height)
        sil = np.zeros_like(v.mask, bool)
        sil[w[ok].astype(int), u[ok].astype(int)] = True
        sil = ndimage.binary_dilation(sil, iterations=splat)
        sil = ndimage.binary_closing(sil, iterations=splat)
        m = v.mask
        inter = (sil & m).sum()
        union = (sil | m).sum()
        ious.append(inter / max(union, 1))
    return float(np.mean(ious))


def _trimmed(d: np.ndarray, keep: float = 0.8) -> np.ndarray:
    k = max(1, int(len(d) * keep))
    return np.sort(d)[:k]


def align_generated(
    gen_pts: np.ndarray,
    obs_pts: np.ndarray,
    views: list[ViewSilhouette],
    support_y: float | None = None,
    yaw_steps: int = 36,
    icp_iters: int = 8,
) -> Alignment:
    """gen_pts: points sampled on the generated surface (canonical, +Y up).
    obs_pts: observed surface points of the object (world, +Y up, metres)."""
    g = gen_pts - (gen_pts.min(0) + gen_pts.max(0)) / 2  # canonical bbox centre at origin
    g_h = float(np.ptp(g[:, 1]))
    o_lo, o_hi = np.percentile(obs_pts, 1, axis=0), np.percentile(obs_pts, 99, axis=0)
    bottom = o_lo[1]
    if support_y is not None and o_lo[1] - support_y < 0.05:
        bottom = support_y  # the object stands on the support: its full height is known
    h_obs = max(float(o_hi[1] - bottom), 1e-3)
    s0 = h_obs / max(g_h, 1e-6)
    size = float(np.linalg.norm(o_hi - o_lo))
    obs = obs_pts if len(obs_pts) <= 6000 else obs_pts[np.random.default_rng(0).choice(len(obs_pts), 6000, replace=False)]

    def fit_translation(R: np.ndarray, s: float, t: np.ndarray) -> tuple[np.ndarray, float]:
        """Translation-only trimmed ICP (observed → generated), y pinned to the support."""
        for _ in range(icp_iters):
            G = (g * s) @ R.T + t
            d, nn = cKDTree(G).query(obs)
            keep = d <= np.percentile(d, 80)
            delta = (obs[keep] - G[nn[keep]]).mean(0)
            t = t + delta
            t[1] = bottom - (g[:, 1].min() * s)  # base on the support
        G = (g * s) @ R.T + t
        d, _ = cKDTree(G).query(obs)
        return t, float(np.sqrt(np.mean(_trimmed(d) ** 2)))

    cands = []
    for k in range(yaw_steps):
        th = 2 * np.pi * k / yaw_steps
        R = _rot_y(th)
        t0 = np.array([obs[:, 0].mean(), bottom - g[:, 1].min() * s0, obs[:, 2].mean()])
        # observed points are the camera-facing half: start with the object pushed back by half its depth
        t, rms = fit_translation(R, s0, t0)
        G = (g * s0) @ R.T + t
        iou = _silhouette_iou(G, views)
        score = rms / max(size, 1e-6) + 0.5 * (1 - iou)
        cands.append({"yaw": th, "t": t, "rms": rms, "iou": iou, "score": score})
    # Refine the three best distinct headings (a half-visible surface often has a near-tie
    # between front/back or side hypotheses) with a fine local search + Procrustes ICP.
    ranked = sorted(cands, key=lambda c: c["score"])
    seeds: list[dict] = []
    for c in ranked:
        if all(abs(np.angle(np.exp(1j * (c["yaw"] - o["yaw"])))) > np.radians(25) for o in seeds):
            seeds.append(c)
        if len(seeds) == 3:
            break

    def evaluate(th: float, s: float, t: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        t, rms = fit_translation(_rot_y(th), s, t)
        iou = _silhouette_iou((g * s) @ _rot_y(th).T + t, views)
        return rms / max(size, 1e-6) + 0.5 * (1 - iou), rms, iou, t

    finals = []
    for seed in seeds:
        th, s, t = seed["yaw"], s0, seed["t"].copy()
        best_local = (seed["score"], seed["rms"], seed["iou"], t, th)
        for dth in np.radians(np.arange(-15, 15.1, 2.5)):
            sc, rms, iou, tt = evaluate(seed["yaw"] + dth, s, t.copy())
            if sc < best_local[0]:
                best_local = (sc, rms, iou, tt, seed["yaw"] + dth)
        _, _, _, t, th = best_local
        for _ in range(icp_iters):
            R = _rot_y(th)
            G = (g * s) @ R.T + t
            d, nn = cKDTree(G).query(obs)
            keep = d <= np.percentile(d, 75)
            A, B = g[nn[keep]], obs[keep]
            a_xz, b_xz = A[:, [0, 2]], B[:, [0, 2]]
            am, bm = a_xz.mean(0), b_xz.mean(0)
            H = (a_xz - am).T @ (b_xz - bm)
            U, S, Vt = np.linalg.svd(H)
            Rot2 = Vt.T @ U.T
            if np.linalg.det(Rot2) < 0:
                Vt[-1] *= -1
                Rot2 = Vt.T @ U.T
            th_new = float(np.arctan2(Rot2[0, 1], Rot2[0, 0]))  # Rot2 = [[c, s], [-s, c]] in (x, z)
            # ICP on a half-visible surface can drift: accept only small heading updates
            if abs(np.angle(np.exp(1j * (th_new - th)))) < np.radians(10):
                th = th_new
            s_xz = S.sum() / max(((a_xz - am) ** 2).sum(), 1e-9)
            s = float(np.clip(0.5 * s + 0.5 * s_xz, 0.85 * s0, 1.15 * s0))  # height anchors scale
            t, _ = fit_translation(_rot_y(th), s, t)
        sc, rms, iou, t = evaluate(th, s, t)
        finals.append((sc, rms, iou, th, s, t))
    score, rms, iou, th, s, t = min(finals, key=lambda f: f[0])
    R = _rot_y(th)

    M = np.eye(4)
    M[:3, :3] = R * s
    M[:3, 3] = t - (R * s) @ ((gen_pts.min(0) + gen_pts.max(0)) / 2)
    reason = ""
    accepted = True
    G = (g * s) @ R.T + t
    gen_foot = float(max(np.ptp(G[:, 0]), np.ptp(G[:, 2])))
    obs_foot = float(max(o_hi[0] - o_lo[0], o_hi[2] - o_lo[2]))
    if views and iou < 0.55:
        accepted, reason = False, f"silhouette agreement too low (IoU {iou:.2f})"
    elif rms > 0.05 * size + 0.005:
        accepted, reason = False, f"generated surface does not match observed geometry (RMS {rms * 100:.1f} cm)"
    elif gen_foot > 3.0 * obs_foot + 0.05:
        # the hidden half can make an object deeper than what is visible, but not 3× larger
        accepted, reason = False, f"generated footprint {gen_foot:.2f} m is implausible for an object {obs_foot:.2f} m wide"
    return Alignment(M, float(np.degrees(th)) % 360, s, rms, iou, score, accepted, reason,
                     [{"yaw_deg": round(float(np.degrees(c["yaw"])), 1), "score": round(c["score"], 4), "iou": round(c["iou"], 3)} for c in cands])
