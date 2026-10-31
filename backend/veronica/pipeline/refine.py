"""Interactive refinement: add objects the detector missed (click or text) and
rebuild the scene geometry from the cached analysis — no depth re-inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..providers.detect import GroundingDinoDetector
from ..providers.registry import inference_lock
from ..providers.segment import Sam2Segmenter
from .analysis import Analysis
from .build import build_scene
from .progress import ProgressFn, null_progress


def add_objects(
    scene_dir: Path,
    *,
    points: list[tuple[float, float]] | None = None,
    point_labels: list[int] | None = None,
    text: str | None = None,
    label: str | None = None,
    progress: ProgressFn = null_progress,
) -> dict:
    an = Analysis.load(scene_dir / "analysis.npz")
    img = Image.fromarray(an.image)
    new_masks, new_labels, new_scores, new_q, src = [], [], [], [], []
    with inference_lock:
        if points:
            progress("segment", 0.2, "Segmenting clicked region")
            m, q = Sam2Segmenter().from_points(img, points, point_labels or [1] * len(points))
            new_masks.append(m), new_labels.append((label or "object").strip().lower()), new_scores.append(1.0), new_q.append(q), src.append("click")
        if text:
            progress("detect", 0.2, f"Searching for “{text}”")
            dets = GroundingDinoDetector()(img, [t.strip() for t in text.split(",") if t.strip()], box_threshold=0.25)
            if dets:
                boxes = np.array([d.box for d in dets], np.float32)
                ms, qs = Sam2Segmenter().from_boxes(img, boxes)
                for d, m, q in zip(dets, ms, qs):
                    new_masks.append(m), new_labels.append(d.label), new_scores.append(d.score), new_q.append(float(q)), src.append("text")
    if not new_masks:
        raise ValueError("Nothing found for that prompt")
    an.masks = np.concatenate([an.masks.reshape(-1, *an.depth.shape), np.stack(new_masks)])
    an.labels += new_labels
    an.scores += [round(float(s), 4) for s in new_scores]
    # user-provided masks win overlaps against everything except smaller ones
    an.mask_quality += [round(max(float(q), 0.51), 4) for q in new_q]
    an.sources += src
    an.save(scene_dir / "analysis.npz")
    progress("geometry", 0.6, "Rebuilding scene")
    import json

    old = json.loads((scene_dir / "manifest.json").read_text())
    man = build_scene(an, scene_dir, old["id"], progress, old["source"].get("name", ""))
    man["added"] = [f"obj-{i}" for i in range(len(an.masks) - len(new_masks), len(an.masks))]
    return man
