"""Open-vocabulary object detection (Grounding DINO).

Grounding DINO's text encoder caps prompts at 256 tokens, so the vocabulary is
queried in chunks and the detections merged with class-agnostic NMS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from ..config import MODELS, settings
from .registry import get_or_load

# Things people photograph indoors. Surfaces that are better modelled as planes
# (floor, wall, ceiling) are deliberately absent: the plane fitter owns those.
DEFAULT_VOCABULARY: list[str] = [
    "table", "desk", "chair", "sofa", "bed", "cabinet", "shelf", "bookshelf", "drawer", "door",
    "window", "curtain", "computer monitor", "television", "laptop", "keyboard", "computer mouse",
    "computer tower", "speaker", "phone", "tablet", "lamp", "bottle", "cup", "mug", "glass",
    "bowl", "plate", "saucer", "jar", "can", "vase", "plant", "potted plant", "book", "notebook", "box", "bag", "backpack",
    "pillow", "blanket", "clock", "picture frame", "poster", "mirror", "power socket", "light switch",
    "remote control", "headphones", "pen", "scissors", "basket", "trash can", "fan", "refrigerator",
    "microwave", "oven", "sink", "stove", "kettle", "toaster", "pan", "pot", "knife", "fork", "spoon",
    "shoe", "toy", "camera", "printer", "router", "cable", "fruit", "candle", "rug", "stool", "bench",
]

CHUNK = 18  # larger prompts measurably lose small objects (bottle, mouse) on the desk test


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


class GroundingDinoDetector:
    name = "grounding-dino-base"

    def _load(self):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        hf = MODELS[self.name].hf_id
        return AutoProcessor.from_pretrained(hf), GroundingDinoForObjectDetection.from_pretrained(hf).to(settings.device).eval()

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        vocabulary: list[str] | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        nms_iou: float = 0.6,
    ) -> list[Detection]:
        proc, model = get_or_load(self.name, self._load)
        vocab = vocabulary or DEFAULT_VOCABULARY
        W, H = image.size
        boxes, scores, labels = [], [], []
        for i in range(0, len(vocab), CHUNK):
            prompt = ". ".join(vocab[i : i + CHUNK]) + "."
            inputs = proc(images=image, text=prompt, return_tensors="pt").to(settings.device)
            out = model(**inputs)
            r = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(H, W)]
            )[0]
            names = r.get("text_labels", r["labels"])
            for b, s, l in zip(r["boxes"].cpu(), r["scores"].cpu(), names):
                l = str(l).strip()
                if not l:
                    continue
                boxes.append(b.tolist())
                scores.append(float(s))
                labels.append(l)
        if not boxes:
            return []
        bt = torch.tensor(boxes, dtype=torch.float32)
        st = torch.tensor(scores, dtype=torch.float32)
        keep = nms(bt, st, nms_iou).tolist()
        dets = [Detection(_canonical(labels[k], vocab), scores[k], tuple(boxes[k])) for k in keep]
        # Drop near-full-frame boxes: they are almost always the room, not an object.
        return [d for d in dets if _area(d.box) < 0.85 * W * H]


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _canonical(label: str, vocab: list[str]) -> str:
    """Grounding DINO sometimes returns merged phrases ("computer monitor keyboard");
    map back to the longest vocabulary entry contained in the phrase."""
    if label in vocab:
        return label
    hits = [v for v in vocab if v in label]
    return max(hits, key=len) if hits else label


def boxes_array(dets: list[Detection]) -> np.ndarray:
    return np.array([d.box for d in dets], dtype=np.float32).reshape(-1, 4)
