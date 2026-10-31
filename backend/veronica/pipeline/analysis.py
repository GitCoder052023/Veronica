"""Stage 1 of single-view reconstruction: run the neural models.

Produces everything the geometric stage needs — metric depth, intrinsics and instance
masks — and nothing else, so the geometry stage is deterministic and testable
without model weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from ..config import MODELS, settings
from ..providers.depth import exif_focal_px, get_depth_provider
from ..providers.detect import GroundingDinoDetector
from ..providers.registry import inference_lock
from ..providers.segment import Sam2Segmenter
from .progress import ProgressFn, Timer, null_progress

try:  # iPhone photos are HEIC by default
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - optional
    pass


@dataclass
class Analysis:
    image: np.ndarray  # (H,W,3) uint8, the working-resolution photo
    depth: np.ndarray  # (H,W) float32 meters
    fx: float
    fy: float
    cx: float
    cy: float
    focal_source: str
    masks: np.ndarray  # (N,H,W) bool
    labels: list[str]
    scores: list[float]  # detector confidence
    mask_quality: list[float]  # SAM predicted IoU
    sources: list[str]  # "detector" | "click" | "text"
    models: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def save(self, path) -> None:
        np.savez_compressed(
            path, image=self.image, depth=self.depth, K=np.array([self.fx, self.fy, self.cx, self.cy]), masks=self.masks,
            meta=np.array([{"labels": self.labels, "scores": self.scores, "mask_quality": self.mask_quality, "sources": self.sources,
                            "focal_source": self.focal_source, "models": self.models, "timings": self.timings}], dtype=object),
        )

    @staticmethod
    def load(path) -> "Analysis":
        z = np.load(path, allow_pickle=True)
        meta = z["meta"][0]
        fx, fy, cx, cy = z["K"].tolist()
        return Analysis(z["image"], z["depth"], fx, fy, cx, cy, meta["focal_source"], z["masks"], meta["labels"], meta["scores"],
                        meta["mask_quality"], meta["sources"], meta["models"], meta["timings"])


def load_image(path_or_file, max_side: int | None = None) -> tuple[Image.Image, float | None]:
    img = Image.open(path_or_file)
    img = ImageOps.exif_transpose(img)
    f_px = exif_focal_px(img, img.size[0])
    img = img.convert("RGB")
    max_side = max_side or settings.max_image_side
    s = max_side / max(img.size)
    if s < 1:
        new = (round(img.size[0] * s), round(img.size[1] * s))
        img = img.resize(new, Image.LANCZOS)
        if f_px:
            f_px *= s
    return img, f_px


def _model_entry(key: str) -> dict:
    m = MODELS[key]
    return {"key": key, "role": m.role, "id": m.hf_id, "license": m.license, "revision": m.revision}


def analyze(
    img: Image.Image,
    focal_px: float | None = None,
    vocabulary: list[str] | None = None,
    progress: ProgressFn = null_progress,
    depth_model: str | None = None,
) -> Analysis:
    timer = Timer()
    depth_p = get_depth_provider(depth_model)
    det, seg = GroundingDinoDetector(), Sam2Segmenter()
    with inference_lock:
        progress("depth", 0.05, f"Estimating metric depth ({depth_p.name})")
        with timer("depth"):
            d = depth_p(img, focal_px)
        progress("detect", 0.35, "Detecting objects (open vocabulary)")
        with timer("detect"):
            dets = det(img, vocabulary)
        progress("segment", 0.55, f"Segmenting {len(dets)} candidates")
        with timer("segment"):
            boxes = np.array([x.box for x in dets], np.float32).reshape(-1, 4)
            masks, iou = seg.from_boxes(img, boxes)
    return Analysis(
        np.asarray(img), d.depth, d.fx, d.fy, d.cx, d.cy, d.focal_source, masks,
        [x.label for x in dets], [round(x.score, 4) for x in dets], [round(float(v), 4) for v in iou], ["detector"] * len(dets),
        [_model_entry(depth_p.name), _model_entry(det.name), _model_entry(seg.name)], timer.timings,
    )
