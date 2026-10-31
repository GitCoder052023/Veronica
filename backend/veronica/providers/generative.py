"""Client for a remote generative completer (see /completer-worker).

The worker runs TRELLIS on a CUDA machine; this side only ships RGBA crops and turns the
returned GLB into a textured mesh. Configure with:

    VERONICA_COMPLETER_URL=http://gpu-box:8790   VERONICA_COMPLETER_TOKEN=…   (token optional)
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass

import httpx
import numpy as np
import trimesh
from PIL import Image

API = "veronica-completer/1"


@dataclass
class GeneratedMesh:
    positions: np.ndarray  # (V,3) canonical, +Y up
    indices: np.ndarray  # (F,3)
    uvs: np.ndarray | None  # (V,2) top-left origin (VERONICA convention)
    texture: Image.Image | None
    colors: np.ndarray | None  # (V,3) uint8 when the GLB has vertex colours instead
    meta: dict


class CompleterClient:
    def __init__(self, url: str | None = None, token: str | None = None, client: httpx.Client | None = None, timeout: float = 600.0) -> None:
        self.url = (url or os.environ.get("VERONICA_COMPLETER_URL", "")).strip().rstrip("/")
        self.token = token if token is not None else (os.environ.get("VERONICA_COMPLETER_TOKEN", "").strip() or None)
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def configured(self) -> bool:
        return bool(self.url)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def health(self) -> dict:
        if not self.configured:
            return {"configured": False, "ok": False, "reason": "VERONICA_COMPLETER_URL is not set"}
        try:
            r = self._client.get(f"{self.url}/v1/health", headers=self._headers(), timeout=5.0)
            r.raise_for_status()
            return {"configured": True, "url": self.url, **r.json()}
        except Exception as e:
            return {"configured": True, "url": self.url, "ok": False, "reason": f"{type(e).__name__}: {e}"}

    def complete(self, crops: list[Image.Image], params: dict | None = None) -> GeneratedMesh:
        if not self.configured:
            raise RuntimeError("No generative completer configured (set VERONICA_COMPLETER_URL)")
        files = []
        for i, im in enumerate(crops):
            buf = io.BytesIO()
            im.convert("RGBA").save(buf, format="PNG")
            files.append(("images", (f"view{i}.png", buf.getvalue(), "image/png")))
        r = self._client.post(f"{self.url}/v1/complete", files=files, data={"params": json.dumps(params or {})}, headers=self._headers())
        if r.status_code != 200:
            raise RuntimeError(f"completer returned {r.status_code}: {r.text[:300]}")
        meta = json.loads(r.headers.get("X-Veronica-Meta", "{}"))
        return glb_to_mesh(r.content, meta)


def glb_to_mesh(data: bytes, meta: dict | None = None) -> GeneratedMesh:
    loaded = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise RuntimeError("completer returned an empty mesh")
    uvs = tex = cols = None
    vis = loaded.visual
    if isinstance(vis, trimesh.visual.texture.TextureVisuals) and vis.uv is not None:
        mat = vis.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if img is not None:
            tex = img.convert("RGB")
            uvs = np.asarray(vis.uv, np.float32).copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]  # trimesh flips glTF V on load; VERONICA textures use top-left origin
    if tex is None:
        try:
            cols = np.asarray(loaded.visual.to_color().vertex_colors)[:, :3].astype(np.uint8)
        except Exception:
            cols = None
    return GeneratedMesh(np.asarray(loaded.vertices, np.float32), np.asarray(loaded.faces, np.uint32), uvs, tex, cols, meta or {})
