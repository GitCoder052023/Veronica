"""Multi-view reconstruction.

Pipeline for N photos of the same place:

1. Depth Anything 3 → per-view depth + camera intrinsics/extrinsics (one forward pass)
2. Depth Pro on the reference view → metric scale for DA3's normalised units
3. Grounding DINO + SAM 2.1 on every view → instance masks
4. Lift every mask to 3D and associate instances *across views* by voxel overlap
5. Per object: TSDF fusion of its masked depths from all views → one observed mesh;
   only what no view saw is completed, and that completion is carved against
   every view's free space (much tighter than single-view carving)
6. Floor / walls from the fused background, gravity-aligned metric world frame

Assets and manifest use the same format as single-view scenes, plus all cameras.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import marching_cubes

from ..assets import write_mesh, write_points
from ..config import MODELS
from ..geometry.camera import Intrinsics, depth_discontinuities
from ..geometry.completion import FLAT_CLASSES, ObjectContext, completer_for
from ..geometry.mesh import Mesh, auto_stride, grid_mesh
from ..geometry.planes import classify_planes, ransac_planes, world_frame
from ..providers import multiview as mvp
from ..providers.depth import get_depth_provider
from ..providers.detect import GroundingDinoDetector
from ..providers.registry import inference_lock
from ..providers.segment import Sam2Segmenter
from .analysis import load_image
from .build import _support_relations, carve_views, resolve_instances
from .progress import ProgressFn, Timer, null_progress

VOXEL_ASSOC = 0.04  # m, instance association grid
MAX_VIEWS = 12


def multiview_status() -> dict:
    ok, why = mvp.available()
    return {"available": ok, "model": f"{mvp.DA3_REPO} ({mvp.DA3_LICENSE})", **({} if ok else {"reason": why})}


@dataclass
class View:
    image: np.ndarray  # (H,W,3)
    depth: np.ndarray  # (H,W) metric
    conf: np.ndarray
    K: Intrinsics
    R_cr: np.ndarray  # camera_from_reference rotation
    t_cr: np.ndarray
    labels: np.ndarray  # instance label map (index into `inst`), -1 none
    inst: list[dict]  # {"label","score","q","mask_index"}
    clean: np.ndarray

    def ref_points(self) -> np.ndarray:
        """All pixels as 3D points in the reference frame (H,W,3)."""
        pc = self.K.unproject(self.depth)
        return (pc - self.t_cr) @ self.R_cr  # R^T (p - t)


def _vox_keys(P: np.ndarray, s: float) -> set:
    q = np.floor(P / s).astype(np.int64)
    return set(map(tuple, np.unique(q, axis=0)))


def _associate(views: list[View]) -> list[dict]:
    """Greedy cross-view instance association by 3D voxel overlap."""
    items = []
    for vi, v in enumerate(views):
        P = v.ref_points()
        for li, ins in enumerate(v.inst):
            m = (v.labels == li) & v.clean
            if m.sum() < 30:
                continue
            pts = P[m]
            if len(pts) > 20000:
                pts = pts[np.random.default_rng(vi * 31 + li).choice(len(pts), 20000, replace=False)]
            items.append({"view": vi, "li": li, "label": ins["label"], "score": ins["score"], "keys": _vox_keys(pts, VOXEL_ASSOC), "n": int(m.sum())})
    items.sort(key=lambda x: -x["n"])
    clusters: list[dict] = []
    for it in items:
        best, best_ov = None, 0.0
        for c in clusters:
            if it["view"] in c["views_li"]:
                continue  # one instance per view per object
            inter = len(it["keys"] & c["keys"])
            ov = inter / max(1, min(len(it["keys"]), len(c["keys"])))
            same = it["label"] == c["label"]
            if ov > (0.25 if same else 0.5) and ov > best_ov:
                best, best_ov = c, ov
        if best is None:
            clusters.append({"label": it["label"], "votes": {it["label"]: it["score"]}, "keys": set(it["keys"]), "views_li": {it["view"]: it["li"]}})
        else:
            best["keys"] |= it["keys"]
            best["views_li"][it["view"]] = it["li"]
            best["votes"][it["label"]] = best["votes"].get(it["label"], 0) + it["score"]
            best["label"] = max(best["votes"], key=best["votes"].get)
    return clusters


def _tsdf_object(views: list[View], cluster: dict, pts_ref: np.ndarray) -> Mesh | None:
    lo, hi = np.percentile(pts_ref, 1, axis=0), np.percentile(pts_ref, 99, axis=0)
    ext = hi - lo
    vs = float(np.clip(ext.max() / 128, 0.003, 0.015))
    lo, hi = lo - 4 * vs, hi + 4 * vs
    dims = np.ceil((hi - lo) / vs).astype(int) + 1
    if np.prod(dims) > 4_000_000:
        vs = float((np.prod(hi - lo) / 4_000_000) ** (1 / 3))
        dims = np.ceil((hi - lo) / vs).astype(int) + 1
    trunc = 3 * vs
    g = np.stack(np.meshgrid(*[lo[i] + vs * np.arange(dims[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    tsdf = np.ones(len(g), np.float32)
    weight = np.zeros(len(g), np.float32)
    for vi, li in cluster["views_li"].items():
        v = views[vi]
        mask = ndimage.binary_dilation(v.labels == li, iterations=2)
        pc = g @ v.R_cr.T + v.t_cr
        u, w_, z = v.K.project(pc)
        ui, wi = np.round(u).astype(int), np.round(w_).astype(int)
        ok = (z > 0.05) & (ui >= 0) & (ui < v.K.width) & (wi >= 0) & (wi < v.K.height)
        idx = np.flatnonzero(ok)
        d = v.depth[wi[idx], ui[idx]]
        m = mask[wi[idx], ui[idx]] & np.isfinite(d) & (d > 0)
        sdf = d - z[idx]
        upd = m & (sdf > -trunc)
        k = idx[upd]
        val = np.clip(sdf[upd] / trunc, -1, 1)
        wnew = v.conf[wi[k], ui[k]].clip(0.1, None)
        tsdf[k] = (tsdf[k] * weight[k] + val * wnew) / (weight[k] + wnew)
        weight[k] += wnew
    wv = weight.reshape(dims) > 0
    # light smoothing of the observed field removes voxel stair-stepping from depth noise
    vol = tsdf.reshape(dims)
    sm = ndimage.gaussian_filter(np.where(wv, vol, 0.0), 0.8)
    norm = ndimage.gaussian_filter(wv.astype(np.float32), 0.8)
    vol = np.where(norm > 0.2, sm / np.maximum(norm, 1e-6), vol).astype(np.float32)
    if wv.sum() < 50 or vol.min() > 0 or vol.max() < 0:
        return None
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(vs, vs, vs), mask=ndimage.binary_erosion(wv, iterations=1) | (wv & (np.abs(vol) < 1)))
    verts = verts + lo
    return Mesh(verts.astype(np.float32), faces[:, ::-1].astype(np.uint32), observed=np.full(len(verts), 255, np.uint8))


def _color_vertices(mesh: Mesh, views: list[View], to_ref: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Colour each vertex from the view that sees it most head-on (with a depth test)."""
    Rw, tw = to_ref  # p_ref = Rw p_world + tw
    P = mesh.positions @ Rw.T + tw
    best = np.full(len(P), -1.0)
    cols = np.full((len(P), 3), 128, np.uint8)
    for v in views:
        pc = P @ v.R_cr.T + v.t_cr
        u, w_, z = v.K.project(pc)
        ui, wi = np.round(u).astype(int), np.round(w_).astype(int)
        ok = (z > 0.05) & (ui >= 0) & (ui < v.K.width) & (wi >= 0) & (wi < v.K.height)
        d = np.full(len(P), np.inf)
        d[ok] = v.depth[wi[ok], ui[ok]]
        vis = ok & (np.abs(d - z) < 0.03 * z + 0.01)
        score = np.where(vis, 1.0 / z, -1.0)
        upd = score > best
        cols[upd] = v.image[wi[upd], ui[upd]]
        best[upd] = score[upd]
    return cols


def _match_frame(im: Image.Image, W: int, H: int) -> Image.Image:
    """Bring a view to the reference frame size without distorting it: centre-crop to the
    reference aspect ratio, then resize (a portrait shot among landscapes stays undistorted;
    DA3 re-estimates intrinsics per view, so the crop is harmless)."""
    if im.size == (W, H):
        return im
    w, h = im.size
    target = W / H
    if w / h > target:
        nw = round(h * target)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    elif w / h < target:
        nh = round(w / target)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im.resize((W, H), Image.LANCZOS)


def reconstruct_multiview(paths: list[Path], out_dir: Path, scene_id: str, progress: ProgressFn = null_progress, vocabulary: list[str] | None = None, source_name: str = "") -> dict:
    timer = Timer()
    paths = paths[:MAX_VIEWS]
    progress("load", 0.02, f"Decoding {len(paths)} photos")
    imgs = [load_image(p)[0] for p in paths]
    W0, H0 = imgs[0].size
    imgs = [_match_frame(im, W0, H0) for im in imgs]

    det, seg = GroundingDinoDetector(), Sam2Segmenter()
    depth_p = get_depth_provider()
    with inference_lock:
        progress("pose", 0.06, f"Estimating camera poses and depth for {len(imgs)} views (Depth Anything 3)")
        with timer("pose"):
            mv = mvp.infer(imgs)
        progress("depth", 0.2, "Metric scale from the reference view")
        with timer("depth"):
            ref = depth_p(imgs[0])
        h, w = mv.size
        d0 = cv2.resize(mv.depth[0], (W0, H0), interpolation=cv2.INTER_LINEAR)
        ok = np.isfinite(ref.depth) & (ref.depth > 0) & (d0 > 1e-4)
        scale = float(np.median(ref.depth[ok] / d0[ok]))
        views_raw = []
        for i, im in enumerate(imgs):
            progress("segment", 0.25 + 0.4 * i / len(imgs), f"Detecting & segmenting view {i + 1}/{len(imgs)}")
            with timer("detect"):
                dets = det(im, vocabulary)
            with timer("segment"):
                boxes = np.array([d.box for d in dets], np.float32).reshape(-1, 4)
                masks, q = seg.from_boxes(im, boxes)
            views_raw.append((dets, masks, q))

    progress("geometry", 0.68, "Fusing views")
    sx, sy = W0 / w, H0 / h
    views: list[View] = []
    for i, (im, (dets, masks, q)) in enumerate(zip(imgs, views_raw)):
        d = cv2.resize(mv.depth[i], (W0, H0), interpolation=cv2.INTER_LINEAR) * scale
        c = cv2.resize(mv.conf[i], (W0, H0), interpolation=cv2.INTER_LINEAR)
        Kd = mv.K[i]
        K = Intrinsics(Kd[0, 0] * sx, Kd[1, 1] * sy, Kd[0, 2] * sx, Kd[1, 2] * sy, W0, H0)
        R_cr, t_cr = mv.w2c[i][:, :3], mv.w2c[i][:, 3] * scale
        labels, kept = resolve_instances(masks, [float(x) for x in q])
        remap = np.full(len(masks) + 1, -1, np.int32)
        inst = []
        for j, k in enumerate(kept):
            remap[k] = j
            inst.append({"label": dets[k].label, "score": dets[k].score, "q": float(q[k])})
        lab = np.where(labels >= 0, remap[np.clip(labels, 0, None)], -1)
        conf_ok = c >= np.percentile(c, 15)
        clean = np.isfinite(d) & (d > 0.05) & ~depth_discontinuities(d) & conf_ok
        views.append(View(np.asarray(im), d.astype(np.float32), c.astype(np.float32), K, R_cr, t_cr, lab, inst, clean))

    with timer("associate"):
        clusters = _associate(views)

    # ---- structure + world frame from fused background
    with timer("planes"):
        bg = []
        for v in views:
            m = (v.labels < 0) & v.clean
            P = v.ref_points()[m]
            if len(P) > 25000:
                P = P[np.random.default_rng(len(bg)).choice(len(P), 25000, replace=False)]
            bg.append(P)
        bgP = np.concatenate(bg)
        planes = ransac_planes(bgP, thresh_rel=0.015) if len(bgP) > 1000 else []
        obj_pts = [views[vi].ref_points()[(views[vi].labels == li) & views[vi].clean] for c in clusters for vi, li in c["views_li"].items()]
        allobj = np.concatenate(obj_pts) if obj_pts else None
        if allobj is not None and len(allobj) > 20000:
            allobj = allobj[np.random.default_rng(0).choice(len(allobj), 20000, replace=False)]
        up = classify_planes(planes, bgP, allobj)
        floor = next((p for p in planes if p.kind in ("floor", "support")), None)
        anchor = np.median(np.concatenate(obj_pts), 0) if obj_pts else np.median(bgP, 0)
        frame = world_frame(up, floor, anchor, [p for p in planes if p.kind == "wall"])
    R, t = frame.R, frame.t  # world_from_ref
    floor_y = 0.0 if floor is not None else None
    to_ref = (R.T, -R.T @ t)
    carve_set = [(v.R_cr @ R.T, v.t_cr - v.R_cr @ R.T @ t, v.K, v.depth) for v in views]

    nodes: list[dict] = []
    objs: list[dict] = []
    with timer("objects"):
        for ci, c in enumerate(clusters):
            label = c["label"]
            progress("geometry", 0.7 + 0.2 * ci / max(1, len(clusters)), f"Fusing {label} from {len(c['views_li'])} view(s)")
            pts_ref = np.concatenate([views[vi].ref_points()[(views[vi].labels == li) & views[vi].clean] for vi, li in c["views_li"].items()])
            if len(pts_ref) < 60:
                continue
            surf_ref = _tsdf_object(views, c, pts_ref)
            if surf_ref is None or surf_ref.n_faces < 8:
                continue
            surf = surf_ref.transformed(R, t)
            surf.colors = _color_vertices(surf, views, to_ref)
            spacing = float(np.median(np.linalg.norm(np.diff(surf.positions[surf.indices[:, :2]], axis=1)[:, 0], axis=1)))
            comp = completer_for(label).complete(ObjectContext(label, surf, t.copy(), floor_y, spacing))
            thick = comp.params.get("depth_m") or 2 * comp.params.get("mean_radius_m", 0.5)
            comp.mesh, carved = carve_views(comp.mesh, carve_set, max_band=max(0.004, 0.35 * float(thick)))
            comp.params["freeSpaceCarved"] = round(carved, 3)
            comp.params["views"] = len(c["views_li"])
            pw = pts_ref @ R.T + t
            pc = np.concatenate([views[vi].image[(views[vi].labels == li) & views[vi].clean] for vi, li in c["views_li"].items()])
            if len(pw) > 150_000:
                sel = np.random.default_rng(ci).choice(len(pw), 150_000, replace=False)
                pw, pc = pw[sel], pc[sel]
            bs = [surf.positions] + ([comp.mesh.positions] if comp.mesh is not None and comp.mesh.n_vertices else [])
            allp = np.concatenate(bs)
            wmin, wmax = allp.min(0), allp.max(0)
            center = (wmin + wmax) / 2
            oid = f"obj-{ci}"
            reps = {
                "surface": write_mesh(out_dir / f"{oid}.surface.vrnc", surf.transformed(np.eye(3), -center)),
                "points": write_points(out_dir / f"{oid}.points.vrnc", pw - center, pc),
            }
            if comp.mesh is not None and comp.mesh.n_faces:
                reps["completion"] = write_mesh(out_dir / f"{oid}.completion.vrnc", comp.mesh.transformed(np.eye(3), -center))
            a_obs = float(surf.face_normals()[1].sum())
            a_inf = float(comp.mesh.face_normals()[1].sum()) if comp.mesh is not None and comp.mesh.n_faces else 0.0
            objs.append({
                "id": oid, "name": label.title(), "label": label, "kind": "object", "flat": label in FLAT_CLASSES,
                "transform": {"position": center.round(5).tolist(), "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]},
                "bounds": {"min": (wmin - center).round(5).tolist(), "max": (wmax - center).round(5).tolist()},
                "dimensions": (wmax - wmin).round(4).tolist(), "representations": reps,
                "detection": {"score": round(max(c["votes"].values()) / max(1, len(c["views_li"])), 4), "maskQuality": 1.0, "source": "detector", "views": sorted(c["views_li"]),
                              "viewLabels": {str(vi): int(li) for vi, li in c["views_li"].items()},
                              "bbox": [0, 0, 0, 0], "pixels": int(len(pts_ref)), "maskIndex": -1},
                "provenance": {"observedFraction": round(a_obs / max(a_obs + a_inf, 1e-9), 3), "completion": comp.method if comp.mesh is not None else "none",
                               "completionParams": comp.params, "fusion": f"TSDF over {len(c['views_li'])} view(s)"},
                "_wmin": wmin.tolist(), "_wmax": wmax.tolist(),
            })
    _support_relations(objs, floor_y, f"structure-{floor.kind}" if floor is not None else "structure-floor")
    for o in objs:
        o.pop("_wmin"), o.pop("_wmax")
    nodes.extend(objs)

    # ---- structure planes: fused background points on each plane → coloured grid
    with timer("structure"):
        for p in [p for p in planes if p.kind in ("floor", "support", "wall", "ceiling")]:
            P_all, C_all = [], []
            for v in views:
                m = (v.labels < 0) & v.clean
                Pr = v.ref_points()[m]
                d = np.abs(Pr @ p.normal + p.d)
                on = d < 0.02 * np.maximum(np.linalg.norm(Pr, axis=1), 0.3)
                P_all.append(Pr[on])
                C_all.append(v.image[m][on])
            Pp = np.concatenate(P_all)
            if len(Pp) < 500:
                continue
            Cp = np.concatenate(C_all)
            n = p.normal
            e1 = np.cross(n, [0, 1, 0] if abs(n[1]) < 0.9 else [1, 0, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            uv = np.stack([Pp @ e1, Pp @ e2], 1)
            cell = 0.03
            lo_uv = np.percentile(uv, 0.5, axis=0)
            hi_uv = np.percentile(uv, 99.5, axis=0)
            gs = np.ceil((hi_uv - lo_uv) / cell).astype(int) + 1
            if gs.prod() > 600_000:
                continue
            ij = np.clip(((uv - lo_uv) / cell).astype(int), 0, gs - 1)
            occ = np.zeros(gs, bool)
            occ[ij[:, 0], ij[:, 1]] = True
            col = np.zeros((*gs, 3), np.float64)
            cnt = np.zeros(gs, np.float64)
            np.add.at(col, (ij[:, 0], ij[:, 1]), Cp)
            np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
            occ = ndimage.binary_closing(occ, iterations=2) | occ
            filled = ndimage.binary_fill_holes(occ)
            observed = cnt > 0
            # colour unobserved cells from the nearest observed cell
            _, (ii, jj) = ndimage.distance_transform_edt(~observed, return_indices=True)
            colg = (col / np.maximum(cnt, 1)[..., None])[ii, jj]
            gi, gj = np.meshgrid(np.arange(gs[0]), np.arange(gs[1]), indexing="ij")
            pts_plane = (lo_uv[0] + gi[..., None] * cell) * e1 + (lo_uv[1] + gj[..., None] * cell) * e2 - p.d * n
            mesh = grid_mesh(pts_plane.astype(np.float32), filled, 1, max_edge_ratio=1e9, observed=observed)
            mesh.uvs = None
            uq = np.clip(np.round((mesh.positions @ e1 - lo_uv[0]) / cell).astype(int), 0, gs[0] - 1)
            vq = np.clip(np.round((mesh.positions @ e2 - lo_uv[1]) / cell).astype(int), 0, gs[1] - 1)
            mesh.colors = colg[uq, vq].astype(np.uint8)
            mesh.texture = None
            wm = mesh.transformed(R, t)
            if wm.n_faces == 0:
                continue
            lo, hi = wm.positions.min(0), wm.positions.max(0)
            cc = (lo + hi) / 2
            same = sum(1 for nd in nodes if nd.get("label") == p.kind)
            sid = f"structure-{p.kind}" + (f"-{same + 1}" if same else "")
            nodes.append({
                "id": sid, "name": p.kind.title() + (f" {same + 1}" if same else ""), "label": p.kind, "kind": "structure",
                "transform": {"position": cc.round(5).tolist(), "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]},
                "bounds": {"min": (lo - cc).round(5).tolist(), "max": (hi - cc).round(5).tolist()}, "dimensions": (hi - lo).round(4).tolist(),
                "representations": {"surface": write_mesh(out_dir / f"{sid}.surface.vrnc", wm.transformed(np.eye(3), -cc))},
                "plane": {"normal": frame.apply_dir(n).round(5).tolist(), "inferredPixels": int((filled & ~observed).sum()), "observedPixels": int(observed.sum())},
                "provenance": {"observedFraction": round(float(observed.sum() / max(filled.sum(), 1)), 3), "completion": "plane hole filling (multi-view)"},
            })

    # ---- textures & cameras
    cams = []
    for i, v in enumerate(views):
        Image.fromarray(v.image).save(out_dir / f"view{i}.jpg", quality=90)
        R_rc = v.R_cr.T
        c_ref = -R_rc @ v.t_cr
        Rw = R @ R_rc
        M = np.eye(4)
        M[:3, :3] = Rw @ np.diag([1.0, -1.0, -1.0])
        M[:3, 3] = R @ c_ref + t
        cams.append({"worldFromCamera": M.round(6).tolist(), "vfov": round(float(np.degrees(2 * np.arctan(H0 / 2 / v.K.fy))), 3), "image": f"view{i}.jpg"})
    # per-view geometry for later refinement (generative completion, carving)
    np.savez_compressed(
        out_dir / "views.npz",
        depth=np.stack([v.depth for v in views]).astype(np.float16),
        K=np.array([[v.K.fx, v.K.fy, v.K.cx, v.K.cy] for v in views]),
        R_cw=np.stack([c[0] for c in carve_set]), t_cw=np.stack([c[1] for c in carve_set]),
        labels=np.stack([v.labels for v in views]).astype(np.int16),
    )
    Image.fromarray(views[0].image).save(out_dir / "image.jpg", quality=92)
    lab0 = views[0].labels
    Image.fromarray(np.where(lab0 >= 0, lab0 + 1, 0).astype(np.uint8)).save(out_dir / "labels.png")
    K0 = views[0].K
    manifest = {
        "version": 1, "id": scene_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"name": source_name, "width": W0, "height": H0, "views": len(views)},
        "mode": "multi-view",
        "camera": {"intrinsics": K0.to_dict(), "hfov": round(float(np.degrees(2 * np.arctan(W0 / 2 / K0.fx))), 3), "vfov": cams[0]["vfov"],
                   "focalSource": "model", "worldFromCamera": cams[0]["worldFromCamera"]},
        "cameras": cams,
        "world": {"up": [0, 1, 0], "units": "meters", "upSource": frame.up_source, "floorY": floor_y,
                  "scaleNote": "Metric scale comes from monocular depth on the reference view (typically within ~10–20%)."},
        "textures": {"image": "image.jpg", "labels": "labels.png", **{f"view{i}": f"view{i}.jpg" for i in range(len(views))}},
        "nodes": nodes,
        "pipeline": {
            "models": [
                {"key": "da3-base", "role": "multi-view pose+depth", "id": mvp.DA3_REPO, "license": mvp.DA3_LICENSE, "revision": None},
                {"key": depth_p.name, "role": "metric scale", "id": MODELS[depth_p.name].hf_id, "license": MODELS[depth_p.name].license, "revision": None},
                {"key": det.name, "role": "detect", "id": MODELS[det.name].hf_id, "license": MODELS[det.name].license, "revision": None},
                {"key": seg.name, "role": "segment", "id": MODELS[seg.name].hf_id, "license": MODELS[seg.name].license, "revision": None},
            ],
            "timings": timer.timings, "metricScale": round(scale, 5),
        },
        "uncertainty": {"notes": [
            "Object surfaces are fused (TSDF) from every view that saw them; only regions no photo observed are inferred.",
            "Inferred geometry is carved against free space observed in all views.",
            "Camera poses come from a feed-forward model; large baselines or low overlap reduce accuracy.",
        ]},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    progress("geometry", 0.99, "Done")
    return manifest
