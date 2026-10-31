"""Feed-forward multi-view geometry: Depth Anything 3 (DA3-BASE, Apache-2.0).

From N unposed photos DA3 predicts per-view depth, confidence, intrinsics and
camera extrinsics in one forward pass — the role DUSt3R/MASt3R play, without
their non-commercial licence or global-alignment optimisation.

We call DA3's network directly instead of its high-level `api` module because that
module eagerly imports export / benchmarking extras (pycolmap, open3d, evo, moviepy)
that inference never touches; two tiny stand-in modules satisfy those imports.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from ..config import settings
from .registry import get_or_load

DA3_MODEL = "da3-base"
DA3_REPO = "depth-anything/DA3-BASE"
DA3_LICENSE = "Apache-2.0"


def _install_import_shims() -> None:
    """Satisfy imports of optional DA3 extras that are not used for inference."""
    if "evo.core.trajectory" not in sys.modules:
        evo = types.ModuleType("evo")
        core = types.ModuleType("evo.core")
        traj = types.ModuleType("evo.core.trajectory")

        class PosePath3D:  # only used when aligning to *given* extrinsics
            def __init__(self, *a, **k):
                raise RuntimeError("pose alignment to input extrinsics is not supported in VERONICA")

        traj.PosePath3D = PosePath3D
        evo.core, core.trajectory = core, traj
        sys.modules.update({"evo": evo, "evo.core": core, "evo.core.trajectory": traj})
    if "depth_anything_3.utils.export" not in sys.modules:
        exp = types.ModuleType("depth_anything_3.utils.export")

        def export(*a, **k):
            raise RuntimeError("DA3 file export is not used by VERONICA")

        exp.export = export
        sys.modules["depth_anything_3.utils.export"] = exp


def available() -> tuple[bool, str]:
    try:
        _install_import_shims()
        import depth_anything_3.api  # noqa: F401
        return True, ""
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"{type(e).__name__}: {e}"


@dataclass
class MultiViewResult:
    depth: np.ndarray  # (N,h,w) float32, DA3 scale (made metric by the caller)
    conf: np.ndarray  # (N,h,w)
    K: np.ndarray  # (N,3,3) at (h,w)
    w2c: np.ndarray  # (N,3,4) OpenCV world→camera
    size: tuple[int, int]  # (h, w)


def _load():
    _install_import_shims()
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(DA3_REPO)
    return model.to(settings.device).eval()


@torch.inference_mode()
def infer(images: list[Image.Image], process_res: int = 504) -> MultiViewResult:
    model = get_or_load(DA3_MODEL, _load)
    pred = model.inference(images, process_res=process_res, ref_view_strategy="first")
    d = pred.depth.astype(np.float32)
    w2c = pred.extrinsics.astype(np.float64)
    if w2c.shape[-2] == 4:
        w2c = w2c[:, :3]
    return MultiViewResult(d, (pred.conf if pred.conf is not None else np.ones_like(d)).astype(np.float32),
                           pred.intrinsics.astype(np.float64), w2c, d.shape[1:])
