"""Promptable instance segmentation (SAM 2.1)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..config import MODELS, settings
from .registry import get_or_load


class Sam2Segmenter:
    name = "sam2.1-hiera-base-plus"

    def _load(self):
        from transformers import Sam2Model, Sam2Processor

        hf = MODELS[self.name].hf_id
        return Sam2Processor.from_pretrained(hf), Sam2Model.from_pretrained(hf).to(settings.device).eval()

    @torch.inference_mode()
    def from_boxes(self, image: Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (masks bool (N,H,W), predicted IoU (N,))."""
        if len(boxes) == 0:
            W, H = image.size
            return np.zeros((0, H, W), bool), np.zeros((0,), np.float32)
        proc, model = get_or_load(self.name, self._load)
        inputs = proc(images=image, input_boxes=[boxes.tolist()], return_tensors="pt").to(settings.device)
        out = model(**inputs, multimask_output=False)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][:, 0]
        return masks.numpy().astype(bool), out.iou_scores.cpu().numpy().reshape(-1).astype(np.float32)

    @torch.inference_mode()
    def from_points(
        self, image: Image.Image, points: list[tuple[float, float]], labels: list[int]
    ) -> tuple[np.ndarray, float]:
        """One object from positive (1) / negative (0) clicks. Picks the best of 3 hypotheses."""
        proc, model = get_or_load(self.name, self._load)
        inputs = proc(
            images=image, input_points=[[[list(p) for p in points]]], input_labels=[[labels]], return_tensors="pt"
        ).to(settings.device)
        out = model(**inputs, multimask_output=True)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0]
        iou = out.iou_scores.cpu().numpy().reshape(-1)
        k = int(np.argmax(iou))
        return masks[k].numpy().astype(bool), float(iou[k])
