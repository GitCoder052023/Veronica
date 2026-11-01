"""Generative completion of an object's unseen side (completer worker: TRELLIS on CUDA, or
Hunyuan3D-2mini on this Mac).

    crops (RGBA, every view that segmented the object)
        → worker → canonical mesh (textured; or bare, then coloured from the photo)
        → align to the object's measured surface + silhouettes (geometry/align.py)
        → carve against every view's observed free space (only unseen parts survive)
        → stored as gen-<id>.{vrnc,png,json}; replaces the symmetry-prior completion

Generated results persist across scene rebuilds and can be reverted to the prior.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageFilter
from scipy import ndimage

from ..assets import read_packet, write_mesh
from ..geometry.align import ViewSilhouette, align_generated
from ..geometry.camera import Intrinsics
from ..geometry.mesh import Mesh
from ..providers.generative import CompleterClient, GeneratedMesh
from .build import carve_views
from .progress import ProgressFn, null_progress

MAX_VIEWS = 4


@dataclass
class SceneView:
    image: np.ndarray
    depth: np.ndarray
    K: Intrinsics
    R_cw: np.ndarray
    t_cw: np.ndarray
    labels: np.ndarray  # instance id map for this view (meaning depends on mode)


def load_scene_views(scene_dir: Path, man: dict) -> list[SceneView]:
    if man["mode"] == "multi-view":
        z = np.load(scene_dir / "views.npz")
        out = []
        for i in range(len(z["depth"])):
            fx, fy, cx, cy = z["K"][i]
            H, W = z["depth"][i].shape
            img = np.asarray(Image.open(scene_dir / f"view{i}.jpg").convert("RGB"))
            out.append(SceneView(img, z["depth"][i].astype(np.float32), Intrinsics(fx, fy, cx, cy, W, H), z["R_cw"][i], z["t_cw"][i], z["labels"][i].astype(np.int32)))
        return out
    a = np.load(scene_dir / "analysis.npz", allow_pickle=True)
    fx, fy, cx, cy = a["K"].tolist()
    H, W = a["depth"].shape
    M = np.array(man["camera"]["worldFromCamera"])
    R_wc = M[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # GL camera → OpenCV camera axes
    labels = np.asarray(Image.open(scene_dir / man["textures"]["labels"])).astype(np.int32)
    return [SceneView(a["image"], a["depth"].astype(np.float32), Intrinsics(fx, fy, cx, cy, W, H), R_wc.T, -R_wc.T @ M[:3, 3], labels)]


def object_masks(node: dict, man: dict, views: list[SceneView]) -> list[tuple[int, np.ndarray]]:
    det = node.get("detection") or {}
    if man["mode"] == "multi-view":
        pairs = [(int(vi), views[int(vi)].labels == int(li)) for vi, li in (det.get("viewLabels") or {}).items()]
    else:
        pairs = [(0, views[0].labels == int(det.get("maskIndex", -1)))]
    pairs = [(vi, m) for vi, m in pairs if m.sum() >= 30]
    return sorted(pairs, key=lambda p: -p[1].sum())[:MAX_VIEWS]


def make_crop(image: np.ndarray, mask: np.ndarray, margin: float = 0.12, max_side: int = 1024) -> Image.Image:
    """RGBA crop with the object mask as (slightly feathered) alpha."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = int(max(x1 - x0, y1 - y0) * margin) + 2
    H, W = mask.shape
    x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    alpha = Image.fromarray((ndimage.binary_dilation(mask, iterations=1)[y0:y1, x0:x1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    rgba = Image.fromarray(image[y0:y1, x0:x1]).convert("RGBA")
    rgba.putalpha(alpha)
    s = max_side / max(rgba.size)
    if s < 1:
        rgba = rgba.resize((round(rgba.size[0] * s), round(rgba.size[1] * s)), Image.LANCZOS)
    elif max(rgba.size) < 256:  # tiny objects: upsample so the generator sees more than a few pixels
        s = 256 / max(rgba.size)
        rgba = rgba.resize((round(rgba.size[0] * s), round(rgba.size[1] * s)), Image.LANCZOS)
    return rgba


def colour_from_photo(P: np.ndarray, view: SceneView, mask: np.ndarray) -> np.ndarray:
    """Vertex colours for a generated shape that came without any (Hunyuan3D-2mini is shape
    only). Each vertex takes the photo colour where it projects inside the object's mask, so
    the hidden side borrows the colour of the visible side in front of it — the same
    assumption the symmetry prior makes — and the nearest mask pixel when it projects
    outside the silhouette. The mask is eroded first so background never bleeds in."""
    core = ndimage.binary_erosion(mask, iterations=2)
    if core.sum() < 10:
        core = mask
    _, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
    Xc = P @ view.R_cw.T + view.t_cw
    z = np.maximum(Xc[:, 2], 1e-6)
    H, W = mask.shape
    u = np.clip(np.round(view.K.fx * Xc[:, 0] / z + view.K.cx), 0, W - 1).astype(int)
    v = np.clip(np.round(view.K.fy * Xc[:, 1] / z + view.K.cy), 0, H - 1).astype(int)
    smooth = ndimage.uniform_filter(view.image.astype(np.float32), size=(3, 3, 1))  # denoise single-pixel picks
    return smooth[iy[v, u], ix[v, u]].round().astype(np.uint8)


def _support_y(node: dict, man: dict) -> float | None:
    sup = (node.get("relations") or {}).get("supportedBy")
    if not sup:
        return None
    other = next((n for n in man["nodes"] if n["id"] == sup), None)
    if other is None:
        return None
    if other["kind"] == "structure":
        return man["world"].get("floorY")
    return float(other["transform"]["position"][1] + other["bounds"]["max"][1])


def _gen_paths(scene_dir: Path, oid: str) -> dict[str, Path]:
    return {k: scene_dir / f"gen-{oid}.{ext}" for k, ext in (("mesh", "vrnc"), ("tex", "png"), ("meta", "json"))}


def generate_completion(scene_dir: Path, oid: str, client: CompleterClient, params: dict | None = None, progress: ProgressFn = null_progress) -> dict:
    man = json.loads((scene_dir / "manifest.json").read_text())
    node = next((n for n in man["nodes"] if n["id"] == oid), None)
    if node is None or node["kind"] != "object":
        raise ValueError(f"{oid} is not an object in this scene")
    views = load_scene_views(scene_dir, man)
    masks = object_masks(node, man, views)
    if not masks:
        raise ValueError("no usable mask for this object")
    progress("generate", 0.1, f"Sending {len(masks)} view(s) of {node['name']} to the generative completer")
    crops = [make_crop(views[vi].image, m) for vi, m in masks]
    gen: GeneratedMesh = client.complete(crops, params)

    progress("align", 0.65, "Aligning generated shape to the measured surface")
    center = np.array(node["transform"]["position"])
    surf = read_packet(scene_dir / node["representations"]["surface"]["file"])
    obs = surf["position"].astype(np.float64) + center
    tm = trimesh.Trimesh(gen.positions, gen.indices, process=False)
    gen_pts = tm.sample(8000)
    sil = [ViewSilhouette(views[vi].K, views[vi].R_cw, views[vi].t_cw, m) for vi, m in masks]
    al = align_generated(gen_pts, obs, sil, support_y=_support_y(node, man))
    paths = _gen_paths(scene_dir, oid)
    meta = {"object": oid, "label": node["label"], "model": gen.meta.get("model", "unknown"), "worker": gen.meta, "views": [vi for vi, _ in masks],
            "crop_sizes": [list(c.size) for c in crops], "alignment": al.to_dict(), "accepted": al.accepted}
    if not al.accepted:
        paths["meta"].write_text(json.dumps(meta, indent=1))
        raise RuntimeError(f"Generated shape rejected: {al.reason}. Kept the symmetry-prior completion.")

    progress("carve", 0.8, "Keeping only what no photo observed")
    P = (np.c_[gen.positions, np.ones(len(gen.positions))] @ al.matrix.T)[:, :3].astype(np.float32)
    colors = gen.colors
    if gen.texture is None and colors is None:
        vi, m = masks[0]
        colors = colour_from_photo(P.astype(np.float64), views[vi], m)
        meta["colour"] = f"from photo (view {vi})"
    mesh = Mesh(P, gen.indices.astype(np.uint32), gen.uvs, colors, np.zeros(len(P), np.uint8), f"gen-{oid}" if gen.texture is not None else None)
    sup = _support_y(node, man)
    if sup is not None:
        below = mesh.positions[:, 1] < sup - 0.01
        mesh.indices = mesh.indices[~below[mesh.indices].any(1)]
        mesh = mesh.compact()
    dims = np.array(node["dimensions"])
    carve_set = [(v.R_cw, v.t_cw, v.K, v.depth) for v in views]
    carved_mesh, carved = carve_views(mesh, carve_set, max_band=max(0.004, 0.35 * float(min(dims[0], dims[2]))))
    if carved_mesh is None:
        raise RuntimeError("Every generated surface coincides with observed geometry; nothing hidden to add.")
    write_mesh(paths["mesh"], carved_mesh)
    if gen.texture is not None:
        gen.texture.save(paths["tex"])
    meta.update({"carved_fraction": round(carved, 3), "faces": carved_mesh.n_faces})
    paths["meta"].write_text(json.dumps(meta, indent=1))
    apply_generated(scene_dir)
    return meta


def apply_generated(scene_dir: Path) -> dict:
    """(Re)apply every accepted generated completion onto the current manifest.
    Called after generation and after any rebuild, so results survive scene rebuilds."""
    man_p = scene_dir / "manifest.json"
    man = json.loads(man_p.read_text())
    for node in man["nodes"]:
        if node["kind"] != "object":
            continue
        paths = _gen_paths(scene_dir, node["id"])
        if not paths["meta"].exists():
            continue
        meta = json.loads(paths["meta"].read_text())
        if not meta.get("accepted") or not paths["mesh"].exists():
            continue
        oid = node["id"]
        center = np.array(node["transform"]["position"])
        prior = scene_dir / f"{oid}.completion.prior.vrnc"
        cur = scene_dir / f"{oid}.completion.vrnc"
        reps = node["representations"]
        if "prior" not in node.get("provenance", {}):
            if cur.exists():
                shutil.copyfile(cur, prior)  # the freshly built symmetry-prior completion
            node["provenance"]["prior"] = {k: node["provenance"].get(k) for k in ("completion", "completionParams", "observedFraction")}
            node["provenance"]["prior"]["representation"] = reps.get("completion")
            node["provenance"]["prior"]["bounds"] = node["bounds"]
        g = read_packet(paths["mesh"])
        m = Mesh(g["position"].astype(np.float32) - center.astype(np.float32), g["index"].reshape(-1, 3).astype(np.uint32),
                 g.get("uv"), g.get("color"), g.get("observed"), f"gen-{oid}" if "uv" in g else None)
        reps["completion"] = write_mesh(cur, m)
        if "uv" in g:
            man["textures"][f"gen-{oid}"] = paths["tex"].name
        surf = read_packet(scene_dir / reps["surface"]["file"])
        a_obs = Mesh(surf["position"], surf["index"].reshape(-1, 3)).face_normals()[1].sum()
        a_inf = m.face_normals()[1].sum()
        lo = np.minimum(np.array(node["bounds"]["min"]), m.positions.min(0))
        hi = np.maximum(np.array(node["bounds"]["max"]), m.positions.max(0))
        node["bounds"] = {"min": lo.round(5).tolist(), "max": hi.round(5).tolist()}
        node["dimensions"] = (hi - lo).round(4).tolist()
        node["provenance"].update({
            "completion": "generative",
            "completionParams": {"model": meta["model"], "views": len(meta["views"]), **meta["alignment"], "freeSpaceCarved": meta.get("carved_fraction"),
                                 "seconds": meta.get("worker", {}).get("seconds")},
            "observedFraction": round(float(a_obs / max(a_obs + a_inf, 1e-9)), 3),
        })
    man_p.write_text(json.dumps(man, indent=1))
    return man


def revert_generated(scene_dir: Path, oid: str) -> dict:
    man_p = scene_dir / "manifest.json"
    man = json.loads(man_p.read_text())
    node = next(n for n in man["nodes"] if n["id"] == oid)
    for p in _gen_paths(scene_dir, oid).values():
        p.unlink(missing_ok=True)
    man["textures"].pop(f"gen-{oid}", None)
    prior = node.get("provenance", {}).pop("prior", None)
    if prior:
        pf = scene_dir / f"{oid}.completion.prior.vrnc"
        if pf.exists():
            shutil.move(pf, scene_dir / f"{oid}.completion.vrnc")
            node["representations"]["completion"] = prior["representation"]
        else:
            node["representations"].pop("completion", None)
        node["bounds"] = prior["bounds"]
        node["dimensions"] = (np.array(prior["bounds"]["max"]) - np.array(prior["bounds"]["min"])).round(4).tolist()
        node["provenance"].update({k: prior[k] for k in ("completion", "completionParams", "observedFraction")})
    man_p.write_text(json.dumps(man, indent=1))
    return man
