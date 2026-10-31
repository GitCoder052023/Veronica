"""Monocular metric depth providers.

A provider returns metric depth (meters, along the optical axis) and pinhole
intrinsics. Depth Pro estimates focal length itself; providers that cannot fall
back to an EXIF focal length or a 55° horizontal FOV, and say so in `focal_source`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from ..config import MODELS, settings
from .registry import get_or_load


@dataclass
class DepthResult:
    depth: np.ndarray  # (H, W) float32 meters
    fx: float
    fy: float
    cx: float
    cy: float
    focal_source: str  # "model" | "exif" | "default"
    model: str


class DepthProvider(Protocol):
    name: str

    def __call__(self, image: Image.Image, exif_focal_px: float | None = None) -> DepthResult: ...


def _dtype() -> torch.dtype:
    return torch.float16 if settings.half_precision and settings.device != "cpu" else torch.float32


class DepthProProvider:
    name = "depth-pro"

    def _load(self):
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor

        hf = MODELS[self.name].hf_id
        proc = DepthProImageProcessor.from_pretrained(hf)
        model = DepthProForDepthEstimation.from_pretrained(hf, dtype=_dtype()).to(settings.device).eval()
        return proc, model

    @torch.inference_mode()
    def __call__(self, image: Image.Image, exif_focal_px: float | None = None) -> DepthResult:
        proc, model = get_or_load(self.name, self._load)
        W, H = image.size
        inputs = proc(images=image, return_tensors="pt").to(settings.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(_dtype())
        out = model(**inputs)
        # When EXIF gives a trustworthy focal length, let Depth Pro rescale depth to it.
        kwargs = {"target_sizes": [(H, W)]}
        if exif_focal_px:
            kwargs["focal_length"] = torch.tensor([exif_focal_px], device=settings.device)
        r = proc.post_process_depth_estimation(out, **kwargs)[0]
        depth = r["predicted_depth"].float().cpu().numpy().astype(np.float32)
        f = float(exif_focal_px) if exif_focal_px else float(r["focal_length"])
        return DepthResult(depth, f, f, W / 2.0, H / 2.0, "exif" if exif_focal_px else "model", self.name)


class DepthAnythingMetricProvider:
    name = "depth-anything-v2-metric-indoor-small"
    default_hfov_deg = 55.0

    def _load(self):
        from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

        hf = MODELS[self.name].hf_id
        return (
            AutoImageProcessor.from_pretrained(hf),
            DepthAnythingForDepthEstimation.from_pretrained(hf).to(settings.device).eval(),
        )

    @torch.inference_mode()
    def __call__(self, image: Image.Image, exif_focal_px: float | None = None) -> DepthResult:
        proc, model = get_or_load(self.name, self._load)
        W, H = image.size
        inputs = proc(images=image, return_tensors="pt").to(settings.device)
        out = model(**inputs)
        r = proc.post_process_depth_estimation(out, target_sizes=[(H, W)])[0]
        depth = r["predicted_depth"].float().cpu().numpy().astype(np.float32)
        if exif_focal_px:
            f, src = float(exif_focal_px), "exif"
        else:
            f, src = (W / 2.0) / math.tan(math.radians(self.default_hfov_deg) / 2.0), "default"
        return DepthResult(depth, f, f, W / 2.0, H / 2.0, src, self.name)


PROVIDERS: dict[str, type] = {p.name: p for p in (DepthProProvider, DepthAnythingMetricProvider)}


def get_depth_provider(name: str | None = None) -> DepthProvider:
    return PROVIDERS[name or settings.depth_model]()


def exif_focal_px(image: Image.Image, width: int) -> float | None:
    """35mm-equivalent focal length from EXIF → pixels. None when absent."""
    try:
        exif = image.getexif().get_ifd(0x8769)
        f35 = exif.get(0xA405)  # FocalLengthIn35mmFilm
        if f35:
            return float(f35) / 36.0 * width
    except Exception:
        pass
    return None
