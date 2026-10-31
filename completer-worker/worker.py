"""VERONICA generative completer worker — runs on a Linux host with an NVIDIA GPU.

Wraps Microsoft TRELLIS (MIT) image-to-3D behind a small HTTP API that the VERONICA
backend calls to generate the unseen side of an object:

    GET  /v1/health                → model, device, VRAM
    POST /v1/complete              → multipart: images (1–4 RGBA PNG crops; alpha = object
                                     mask), params (JSON) → model/gltf-binary (+Y up)

Each image is one *view* of the same object; with several, TRELLIS' multi-image
conditioning is used. Alpha is honoured directly, so no background removal model runs.

Run:  VERONICA_COMPLETER_TOKEN=secret uvicorn worker:app --host 0.0.0.0 --port 8790
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
from typing import Protocol

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

API_VERSION = "veronica-completer/1"
MAX_IMAGES = 4
MAX_BYTES = 12 * 1024 * 1024


class Params(BaseModel):
    seed: int = 1
    ss_steps: int = Field(12, ge=4, le=50)
    ss_cfg: float = Field(7.5, ge=0, le=15)
    slat_steps: int = Field(12, ge=4, le=50)
    slat_cfg: float = Field(3.0, ge=0, le=15)
    simplify: float = Field(0.95, ge=0.5, le=0.99)
    texture_size: int = Field(1024, ge=256, le=2048)
    multi_algo: str = Field("stochastic", pattern="^(stochastic|multidiffusion)$")


class Engine(Protocol):
    name: str

    def info(self) -> dict: ...

    def generate(self, images: list[Image.Image], p: Params) -> bytes: ...


class TrellisEngine:
    """The real thing. Imported lazily so the HTTP layer can be tested without CUDA."""

    def __init__(self, model_id: str | None = None) -> None:
        os.environ.setdefault("SPCONV_ALGO", "native")
        import torch
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils

        self.torch = torch
        self.post = postprocessing_utils
        self.name = model_id or os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-image-large")
        self.pipe = TrellisImageTo3DPipeline.from_pretrained(self.name)
        self.pipe.cuda()

    def info(self) -> dict:
        t = self.torch
        props = t.cuda.get_device_properties(0)
        return {"model": self.name, "license": "MIT", "device": props.name, "vram_gb": round(props.total_memory / 2**30, 1)}

    def generate(self, images: list[Image.Image], p: Params) -> bytes:
        ss = {"steps": p.ss_steps, "cfg_strength": p.ss_cfg}
        sl = {"steps": p.slat_steps, "cfg_strength": p.slat_cfg}
        if len(images) == 1:
            out = self.pipe.run(images[0], seed=p.seed, formats=["gaussian", "mesh"], sparse_structure_sampler_params=ss, slat_sampler_params=sl)
        else:
            out = self.pipe.run_multi_image(images, seed=p.seed, formats=["gaussian", "mesh"], sparse_structure_sampler_params=ss,
                                            slat_sampler_params=sl, mode=p.multi_algo)
        glb = self.post.to_glb(out["gaussian"][0], out["mesh"][0], simplify=p.simplify, texture_size=p.texture_size, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".glb") as f:
            glb.export(f.name)
            data = open(f.name, "rb").read()
        self.torch.cuda.empty_cache()
        return data


def create_app(engine_factory=TrellisEngine) -> FastAPI:
    app = FastAPI(title="VERONICA completer worker", version=API_VERSION)
    state: dict = {"engine": None, "error": None}
    lock = threading.Lock()  # one generation at a time per GPU
    token = os.environ.get("VERONICA_COMPLETER_TOKEN")

    def engine() -> Engine:
        if state["engine"] is None:
            with lock:
                if state["engine"] is None:
                    try:
                        state["engine"] = engine_factory()
                    except Exception as e:  # surfaced through /v1/health
                        state["error"] = f"{type(e).__name__}: {e}"
                        raise HTTPException(503, f"model failed to load: {state['error']}")
        return state["engine"]

    def auth(authorization: str | None = Header(None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong bearer token")

    @app.get("/v1/health")
    def health(_: None = Depends(auth)) -> dict:
        try:
            info = engine().info()
            return {"ok": True, "api": API_VERSION, **info}
        except HTTPException as e:
            return {"ok": False, "api": API_VERSION, "error": e.detail}

    @app.post("/v1/complete")
    async def complete(images: list[UploadFile] = File(...), params: str = Form("{}"), _: None = Depends(auth)) -> Response:
        try:
            p = Params(**json.loads(params or "{}"))
        except (ValidationError, json.JSONDecodeError) as e:
            raise HTTPException(422, f"bad params: {e}")
        if not 1 <= len(images) <= MAX_IMAGES:
            raise HTTPException(422, f"send 1–{MAX_IMAGES} images")
        ims = []
        for f in images:
            data = await f.read()
            if len(data) > MAX_BYTES:
                raise HTTPException(413, "image too large")
            try:
                im = Image.open(io.BytesIO(data))
                im.load()
            except Exception:
                raise HTTPException(415, "not an image")
            if im.mode != "RGBA":
                raise HTTPException(422, "images must be RGBA with the object mask in alpha")
            ims.append(im)
        eng = engine()
        t0 = time.perf_counter()
        with lock:
            glb = eng.generate(ims, p)
        meta = {"api": API_VERSION, "model": eng.name, "seconds": round(time.perf_counter() - t0, 2), "views": len(ims), "params": p.model_dump()}
        return Response(glb, media_type="model/gltf-binary", headers={"X-Veronica-Meta": json.dumps(meta)})

    return app


app = create_app()
