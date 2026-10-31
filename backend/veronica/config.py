"""Runtime configuration and the pinned model registry.

Every model the pipeline can load is listed here with its role, Hugging Face id,
license and a short note, so the manifest of each reconstruction can record exactly
which weights produced it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(frozen=True)
class ModelSpec:
    role: str
    hf_id: str
    license: str
    revision: str | None = None
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "depth-pro": ModelSpec(
        "depth", "apple/DepthPro-hf", "apple-amlr (Apple ML Research Model License: research / non-commercial)",
        note="Metric depth + focal length from a single image.",
    ),
    "depth-anything-v2-metric-indoor-small": ModelSpec(
        "depth", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "Apache-2.0",
        note="Fast metric depth fallback; no focal estimate (uses EXIF or 55° FOV).",
    ),
    "grounding-dino-base": ModelSpec(
        "detect", "IDEA-Research/grounding-dino-base", "Apache-2.0",
        note="Open-vocabulary detection from text prompts.",
    ),
    "sam2.1-hiera-base-plus": ModelSpec(
        "segment", "facebook/sam2.1-hiera-base-plus", "Apache-2.0",
        note="Box / point prompted instance masks.",
    ),
}


def _env(name: str, default: str = "") -> str:
    """Environment value with blank treated as unset (so `KEY=` in .env means default)."""
    v = os.environ.get(name, "").strip()
    return v or default


def _pick_device() -> str:
    forced = _env("VERONICA_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("VERONICA_DATA", str(Path(__file__).resolve().parents[1] / "data"))).expanduser())
    device: str = field(default_factory=_pick_device)
    depth_model: str = field(default_factory=lambda: _env("VERONICA_DEPTH_MODEL", "depth-pro"))
    detect_model: str = "grounding-dino-base"
    segment_model: str = "sam2.1-hiera-base-plus"
    max_image_side: int = int(_env("VERONICA_MAX_SIDE", "1280"))
    half_precision: bool = _env("VERONICA_FP32", "0") != "1"

    @property
    def scenes_dir(self) -> Path:
        p = self.data_dir / "scenes"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
