"""Stage 2 of single-view reconstruction: deterministic geometry.

Turns an `Analysis` (photo, metric depth, intrinsics, instance masks) into a scene:

1. resolve overlapping masks into one instance label map
2. lift pixels to metric 3D, drop flying pixels at occlusion edges
3. find floor / walls with RANSAC and build a gravity-aligned, metric world frame
4. per object: observed surface mesh (full-resolution texture), point cloud and an
   explicitly *inferred* completion of the hidden side
5. structure: floor / wall planes, extended behind objects by ray–plane intersection
   (rooms are convex from the inside, so the first plane hit is what lies behind)
6. leftover background surface, with object holes filled by inpainting (inferred)

Everything is written as VRNC packets plus a JSON manifest.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from ..assets import write_mesh, write_points
from ..geometry.camera import Intrinsics, depth_discontinuities
from ..geometry.completion import FLAT_CLASSES, ObjectContext, completer_for
from ..geometry.mesh import Mesh, auto_stride, grid_mesh
from ..geometry.planes import Plane, classify_planes, ransac_planes, world_frame
from .analysis import Analysis
from .progress import ProgressFn, Timer, null_progress

MANIFEST_VERSION = 1
OBJECT_VERTEX_BUDGET = 180_000
STRUCTURE_VERTEX_BUDGET = 90_000
BACKGROUND_VERTEX_BUDGET = 160_000
POINT_BUDGET = 120_000


# ----------------------------------------------------------------------------- masks
def resolve_instances(
    masks: np.ndarray, quality: list[float], min_quality: float = 0.5, min_area_frac: float = 0.0004, min_keep: float = 0.3
) -> tuple[np.ndarray, list[int]]:
    """Paint masks largest-first so smaller (usually nearer, occluding) instances win
    shared pixels. Returns the label map (-1 = none) and the kept mask indices."""
    N, H, W = masks.shape if masks.ndim == 3 else (0, *masks.shape[-2:])
    labels = np.full((H, W), -1, np.int32)
    if N == 0:
        return labels, []
    areas = masks.reshape(N, -1).sum(1)
    min_area = max(60, min_area_frac * H * W)
    order = [k for k in np.argsort(-areas) if areas[k] >= min_area and quality[k] >= min_quality]
    for k in order:
        m = masks[k]
        # keep only connected parts that matter (SAM sometimes adds specks)
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            m = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.08 * sizes.max()))
        labels[m] = k
    kept = []
    for k in order:
        if (labels == k).sum() >= min_keep * areas[k]:
            kept.append(int(k))
        else:
            labels[labels == k] = -1  # mostly swallowed by others → duplicate
    return labels, kept


# ----------------------------------------------------------------------------- helpers
def _flying_pixel_mask(depth: np.ndarray, rel: float = 0.03) -> np.ndarray:
    med = ndimage.median_filter(depth, size=5)
    return np.abs(depth - med) / np.maximum(depth, 1e-4) < rel


def _ray_plane_depth(rays: np.ndarray, plane: Plane) -> np.ndarray:
    """Depth (z) where each ray (z=1 normalised) meets the plane; inf if parallel/behind."""
    denom = rays @ plane.normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -plane.d / denom
    t[(denom == 0) | ~np.isfinite(t) | (t <= 0)] = np.inf
    return t


def _support_relations(objs: list[dict], floor_y: float | None, floor_id: str = "structure-floor") -> None:
    """Annotate each object with what it rests on (another object or the floor)."""
    for a in objs:
        lo_a, hi_a = np.array(a["_wmin"]), np.array(a["_wmax"])
        best, best_gap = None, 0.06
        for b in objs:
            if b is a:
                continue
            lo_b, hi_b = np.array(b["_wmin"]), np.array(b["_wmax"])
            ox = max(0.0, min(hi_a[0], hi_b[0]) - max(lo_a[0], lo_b[0]))
            oz = max(0.0, min(hi_a[2], hi_b[2]) - max(lo_a[2], lo_b[2]))
            foot = max((hi_a[0] - lo_a[0]) * (hi_a[2] - lo_a[2]), 1e-6)
            gap = abs(lo_a[1] - hi_b[1])
            if ox * oz / foot > 0.3 and gap < best_gap and hi_b[1] < hi_a[1]:
                best, best_gap = b["id"], gap
        if best is None and floor_y is not None and abs(lo_a[1] - floor_y) < 0.06:
            best = floor_id
        a["relations"] = {"supportedBy": best}


def _release_see_through(labels: np.ndarray, kept: list[int], edges: np.ndarray, valid: np.ndarray, pts: np.ndarray, planes: list[Plane]) -> int:
    if not planes:
        return 0
    released = 0
    tol = 0.025 * np.maximum(pts[..., 2], 0.3)
    for k in kept:
        m = (labels == k) & valid
        comp, n = ndimage.label(m & ~edges)
        if n < 2:
            continue
        sizes = ndimage.sum(np.ones_like(m), comp, range(1, n + 1))
        main = int(np.argmax(sizes)) + 1
        for c in range(1, n + 1):
            if c == main:
                continue
            sel = comp == c
            P = pts[sel]
            for p in planes:
                if np.median(np.abs(P @ p.normal + p.d) / tol[sel]) < 1.0:
                    # grow by a pixel to include the edge pixels around the patch
                    grow = ndimage.binary_dilation(sel, iterations=1) & (labels == k) & edges
                    labels[sel | grow] = -1
                    released += int(sel.sum())
                    break
    return released


def carve_views(mesh: Mesh | None, views: list[tuple[np.ndarray, np.ndarray, Intrinsics, np.ndarray]], rel_margin: float = 0.015, max_band: float = 1.0) -> tuple[Mesh | None, float]:
    """Remove inferred geometry that any camera would have seen.

    Each photo proves the space between its camera and the observed surface is empty.
    An inferred vertex that projects into a view *in front of* (or onto) the observed
    depth contradicts that observation — or duplicates observed geometry — so faces
    touching such vertices are dropped. `views` holds (R_cw, t_cw, K, depth) with
    p_cam = R_cw @ p_world + t_cw. Returns the carved mesh and the removed face fraction.
    """
    if mesh is None or mesh.n_faces == 0:
        return mesh, 0.0
    inferred = mesh.observed == 0 if mesh.observed is not None else np.ones(mesh.n_vertices, bool)
    bad = np.zeros(mesh.n_vertices, bool)
    for R_cw, t_cw, K, depth in views:
        pc = mesh.positions @ R_cw.T + t_cw
        u, v, z = K.project(pc)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (z > 0.05) & (ui >= 0) & (ui < K.width) & (vi >= 0) & (vi < K.height)
        zi = depth[np.clip(vi, 0, K.height - 1), np.clip(ui, 0, K.width - 1)]
        band = np.minimum(rel_margin * zi + 0.01, max_band)
        bad |= inside & np.isfinite(zi) & (zi > 0) & (z < zi + band) & inferred
    keep = ~bad[mesh.indices].any(1)
    removed = 1.0 - keep.mean()
    m = Mesh(**{**mesh.__dict__})
    m.indices = mesh.indices[keep]
    m = m.compact()
    return (m if m.n_faces else None), float(removed)


def carve_free_space(mesh: Mesh | None, R: np.ndarray, t: np.ndarray, K: Intrinsics, depth: np.ndarray, rel_margin: float = 0.015, max_band: float = 1.0) -> tuple[Mesh | None, float]:
    """Single-view carving; (R, t) is world_from_camera."""
    return carve_views(mesh, [(R.T, -R.T @ t, K, depth)], rel_margin, max_band)


def _bounds(m: Mesh | None) -> tuple[np.ndarray, np.ndarray] | None:
    if m is None or m.n_vertices == 0:
        return None
    return m.positions.min(0), m.positions.max(0)


# ----------------------------------------------------------------------------- main
def build_scene(an: Analysis, out_dir: Path, scene_id: str, progress: ProgressFn = null_progress, source_name: str = "") -> dict:
    timer = Timer()
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = an.depth.shape
    K = Intrinsics(an.fx, an.fy, an.cx, an.cy, W, H)
    rays = K.rays()
    depth = an.depth.astype(np.float32)
    # Isolated depth outliers ("flying pixels" not on a real edge) take their local median,
    # so thin dark structures keep a continuous surface instead of becoming holes.
    med = ndimage.median_filter(depth, size=5)
    depth = np.where(np.abs(depth - med) / np.maximum(depth, 1e-4) > 0.03, med, depth).astype(np.float32)
    pts_cam = rays * depth[..., None]
    valid_depth = np.isfinite(depth) & (depth > 0.05) & (depth < 80)

    progress("geometry", 0.62, "Resolving instance overlaps")
    with timer("instances"):
        labels, kept = resolve_instances(an.masks, an.mask_quality)
    edges = depth_discontinuities(depth)
    clean = valid_depth & _flying_pixel_mask(depth) & ~edges

    # ---- structure planes & world frame
    progress("geometry", 0.66, "Fitting floor and walls")
    with timer("planes"):
        bg_pix = np.flatnonzero(((labels < 0) & clean).ravel())
        rng = np.random.default_rng(0)
        sub = bg_pix if len(bg_pix) <= 60_000 else rng.choice(bg_pix, 60_000, replace=False)
        P_sub = pts_cam.reshape(-1, 3)[sub]
        planes = ransac_planes(P_sub) if len(sub) > 1000 else []
        obj_sub = np.flatnonzero(((labels >= 0) & clean).ravel())
        if len(obj_sub) > 20_000:
            obj_sub = rng.choice(obj_sub, 20_000, replace=False)
        up = classify_planes(planes, P_sub, pts_cam.reshape(-1, 3)[obj_sub])
        floor = next((p for p in planes if p.kind in ("floor", "support")), None)
        # See-through regions inside a mask (the floor seen between table legs) are cut off
        # from the object by depth discontinuities and lie on a room plane: give them back.
        released = _release_see_through(labels, kept, edges, valid_depth, pts_cam, [p for p in planes if p.kind in ("floor", "support", "wall", "ceiling")])
        obj_pix = (labels >= 0) & clean
        anchor = np.median(pts_cam[obj_pix], 0) if obj_pix.sum() > 50 else np.median(pts_cam[clean], 0)
        frame = world_frame(up, floor, anchor, [p for p in planes if p.kind == "wall"])
    R, t = frame.R, frame.t
    cam_pos = t.copy()
    floor_y = 0.0 if floor is not None else None

    # source textures
    Image.fromarray(an.image).save(out_dir / "image.jpg", quality=93)
    hidden = np.isin(labels, kept)
    hidden_d = cv2.dilate(hidden.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    with timer("inpaint"):
        bg_img = cv2.inpaint(np.ascontiguousarray(an.image[..., ::-1]), hidden_d.astype(np.uint8) * 255, 6, cv2.INPAINT_TELEA)[..., ::-1]
    Image.fromarray(bg_img).save(out_dir / "background.jpg", quality=90)
    Image.fromarray(np.where(labels >= 0, labels + 1, 0).astype(np.uint8)).save(out_dir / "labels.png")

    nodes: list[dict] = []

    # ---- objects
    obj_records: list[dict] = []
    with timer("objects"):
        for i, k in enumerate(kept):
            label = an.labels[k]
            progress("geometry", 0.70 + 0.2 * i / max(len(kept), 1), f"Reconstructing {label}")
            m_k = labels == k
            g_mask = ndimage.binary_erosion(m_k, iterations=1) & clean
            n_pix = int(g_mask.sum())
            if n_pix < 40:
                continue
            stride = auto_stride(n_pix, OBJECT_VERTEX_BUDGET)
            surf_cam = grid_mesh(pts_cam, g_mask, stride, observed=np.ones_like(g_mask), texture="image")
            if surf_cam.n_faces < 8:
                continue
            surf = surf_cam.transformed(R, t)
            z_med = float(np.median(depth[g_mask]))
            spacing = z_med * stride / an.fx
            comp = completer_for(label).complete(ObjectContext(label, surf, cam_pos, floor_y, spacing))
            # small / thin objects get a tighter band so their back side is not mistaken for a duplicate
            thick = comp.params.get("depth_m") or 2 * comp.params.get("mean_radius_m", 0.5)
            comp.mesh, carved = carve_free_space(comp.mesh, R, t, K, depth, max_band=max(0.004, 0.35 * float(thick)))
            comp.params["freeSpaceCarved"] = round(carved, 3)
            # points
            pix = np.flatnonzero(g_mask.ravel())
            if len(pix) > POINT_BUDGET:
                pix = rng.choice(pix, POINT_BUDGET, replace=False)
            pw = frame.apply(pts_cam.reshape(-1, 3)[pix])
            pc = an.image.reshape(-1, 3)[pix]
            # pivot: centre of the full (observed + inferred) bounds
            bs = [b for b in (_bounds(surf), _bounds(comp.mesh)) if b is not None]
            wmin = np.min([b[0] for b in bs], 0)
            wmax = np.max([b[1] for b in bs], 0)
            center = (wmin + wmax) / 2
            oid = f"obj-{k}"
            reps = {
                "surface": write_mesh(out_dir / f"{oid}.surface.vrnc", surf.transformed(np.eye(3), -center)),
                "points": write_points(out_dir / f"{oid}.points.vrnc", pw - center, pc),
            }
            if comp.mesh is not None and comp.mesh.n_faces:
                reps["completion"] = write_mesh(out_dir / f"{oid}.completion.vrnc", comp.mesh.transformed(np.eye(3), -center))
            ys, xs = np.nonzero(m_k)
            n_obs_area = float(surf.face_normals()[1].sum())
            n_inf_area = float(comp.mesh.face_normals()[1].sum()) if comp.mesh is not None and comp.mesh.n_faces else 0.0
            rec = {
                "id": oid, "name": label.title(), "label": label, "kind": "object",
                "flat": label in FLAT_CLASSES,
                "transform": {"position": center.round(5).tolist(), "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]},
                "bounds": {"min": (wmin - center).round(5).tolist(), "max": (wmax - center).round(5).tolist()},
                "dimensions": (wmax - wmin).round(4).tolist(),
                "representations": reps,
                "detection": {"score": an.scores[k], "maskQuality": an.mask_quality[k], "source": an.sources[k],
                              "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "pixels": int(m_k.sum()), "maskIndex": k + 1},
                "provenance": {
                    "observedFraction": round(n_obs_area / max(n_obs_area + n_inf_area, 1e-9), 3),
                    "completion": comp.method, "completionParams": comp.params,
                    "meanDepth": round(z_med, 3),
                },
                "_wmin": wmin.tolist(), "_wmax": wmax.tolist(),
            }
            obj_records.append(rec)
    _support_relations(obj_records, floor_y, f"structure-{floor.kind}" if floor is not None else "structure-floor")
    for r in obj_records:
        r.pop("_wmin"), r.pop("_wmax")
    nodes.extend(obj_records)

    # ---- structure planes (observed pixels + ray–plane continuation behind objects)
    progress("geometry", 0.91, "Building room structure")
    with timer("structure"):
        struct_planes = [p for p in planes if p.kind in ("floor", "support", "wall", "ceiling")]
        bg_mask = (labels < 0) & valid_depth
        plane_label = np.full((H, W), -1, np.int32)
        plane_depth = np.full((H, W), np.inf, np.float32)
        tol = 0.02 * np.maximum(depth, 0.3)
        best_res = np.full((H, W), np.inf, np.float32)
        for j, p in enumerate(struct_planes):
            res = np.abs(pts_cam @ p.normal + p.d)
            win = bg_mask & (res < tol) & (res < best_res)
            plane_label[win] = j
            best_res[win] = res[win]
        # clean speckle per plane
        for j in range(len(struct_planes)):
            m = ndimage.binary_opening(plane_label == j, iterations=2)
            plane_label[(plane_label == j) & ~m] = -1
        # continuation behind objects: first structural plane hit beyond the object
        hit_depth = np.full((H, W), np.inf, np.float32)
        hit_label = np.full((H, W), -1, np.int32)
        for j, p in enumerate(struct_planes):
            dj = _ray_plane_depth(rays, p).astype(np.float32)
            better = hidden_d & (dj > depth * 0.98) & (dj < hit_depth) & (dj < 60)
            hit_depth[better] = dj[better]
            hit_label[better] = j
        inferred_plane = hidden_d & (hit_label >= 0)
        plane_label[inferred_plane] = hit_label[inferred_plane]
        for j, p in enumerate(struct_planes):
            pl = plane_label == j
            if pl.sum() < 200:
                continue
            dj = _ray_plane_depth(rays, p)
            snapped = rays * np.where(np.isfinite(dj), dj, 0)[..., None]
            stride = auto_stride(int(pl.sum()), STRUCTURE_VERTEX_BUDGET)
            mesh = grid_mesh(snapped, pl & np.isfinite(dj), stride, max_edge_ratio=1.0, observed=~(inferred_plane & hidden), texture="background")
            if mesh.n_faces == 0:
                continue
            wm = mesh.transformed(R, t)
            lo, hi = wm.positions.min(0), wm.positions.max(0)
            c = (lo + hi) / 2
            same = sum(1 for n in nodes if n.get("label") == p.kind)
            sid = f"structure-{p.kind}" + (f"-{same + 1}" if same else "")
            n_world = frame.apply_dir(p.normal)
            nodes.append({
                "id": sid, "name": p.kind.title() + (f" {same + 1}" if same else ""), "label": p.kind, "kind": "structure",
                "transform": {"position": c.round(5).tolist(), "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]},
                "bounds": {"min": (lo - c).round(5).tolist(), "max": (hi - c).round(5).tolist()}, "dimensions": (hi - lo).round(4).tolist(),
                "representations": {"surface": write_mesh(out_dir / f"{sid}.surface.vrnc", wm.transformed(np.eye(3), -c))},
                "plane": {"normal": n_world.round(5).tolist(), "inferredPixels": int((pl & inferred_plane & hidden).sum()), "observedPixels": int((pl & ~(inferred_plane & hidden)).sum())},
                "provenance": {"observedFraction": round(float((pl & ~(inferred_plane & hidden)).sum() / max(pl.sum(), 1)), 3), "completion": "ray-plane continuation"},
            })

        # ---- leftover background (not object, not plane), holes inpainted
        rest = (plane_label < 0) & valid_depth
        rest_hidden = rest & hidden_d
        if rest_hidden.any():
            d_fill = depth.copy()
            d_fill[hidden_d] = 0
            d_inp = cv2.inpaint(d_fill, rest_hidden.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
            d_bg = np.where(rest_hidden, d_inp, depth)
        else:
            d_bg = depth
        rest &= d_bg > 0.05
        rest_ok = rest & (rest_hidden | (_flying_pixel_mask(d_bg) & ~edges))
        if rest_ok.sum() > 200:
            stride = auto_stride(int(rest_ok.sum()), BACKGROUND_VERTEX_BUDGET)
            mesh = grid_mesh(rays * d_bg[..., None], rest_ok, stride, observed=~(rest_hidden & hidden), texture="background")
            if mesh.n_faces:
                wm = mesh.transformed(R, t)
                lo, hi = wm.positions.min(0), wm.positions.max(0)
                c = (lo + hi) / 2
                nodes.append({
                    "id": "background", "name": "Background", "label": "background", "kind": "background",
                    "transform": {"position": c.round(5).tolist(), "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]},
                    "bounds": {"min": (lo - c).round(5).tolist(), "max": (hi - c).round(5).tolist()}, "dimensions": (hi - lo).round(4).tolist(),
                    "representations": {"surface": write_mesh(out_dir / "background.surface.vrnc", wm.transformed(np.eye(3), -c))},
                    "provenance": {"observedFraction": round(float((rest_ok & ~rest_hidden).sum() / max(rest_ok.sum(), 1)), 3), "completion": "telea depth inpainting"},
                })

    hfov = float(np.degrees(2 * np.arctan(W / 2 / an.fx)))
    vfov = float(np.degrees(2 * np.arctan(H / 2 / an.fy)))
    manifest = {
        "version": MANIFEST_VERSION,
        "id": scene_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"name": source_name, "width": W, "height": H, "views": 1},
        "mode": "single-view",
        "camera": {
            "intrinsics": K.to_dict(), "hfov": round(hfov, 3), "vfov": round(vfov, 3), "focalSource": an.focal_source,
            "worldFromCamera": frame.camera_matrix_gl().round(6).tolist(),
        },
        "world": {"up": [0, 1, 0], "units": "meters", "upSource": frame.up_source, "floorY": floor_y,
                  "scaleNote": "Metric scale is a monocular estimate (typically within ~10–20%)."},
        "textures": {"image": "image.jpg", "background": "background.jpg", "labels": "labels.png"},
        "nodes": nodes,
        "pipeline": {"models": an.models, "timings": {**an.timings, **timer.timings}},
        "uncertainty": {
            "notes": [
                "Only camera-facing surfaces are observed in a single photo; hidden sides are inferred from symmetry priors and rendered as such.",
                "Background behind objects is filled by plane continuation (floor/walls) or inpainting and marked inferred.",
            ]
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if any(out_dir.glob("gen-*.json")):  # keep generated completions across rebuilds
        from .generate import apply_generated

        manifest = apply_generated(out_dir)
    return manifest
