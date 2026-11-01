"""VERONICA generative completer worker.

Generates the unseen side of an object behind a small HTTP API that the VERONICA backend
calls. Two engines speak the same API:

    trellis          Microsoft TRELLIS (MIT) on a Linux host with an NVIDIA GPU. Textured,
                     multi-view conditioning.
    hunyuan3d-mini   Tencent Hunyuan3D-2mini shape model in pure PyTorch, so it runs on an
                     Apple Silicon Mac (MPS), on CUDA, or (slowly) on CPU. Shape only: the
                     VERONICA backend colours it from the photo. Single-image conditioning.

VERONICA_COMPLETER_ENGINE picks one; the default (auto) uses whichever is installed,
preferring TRELLIS.

    GET  /v1/health                → model, licence, device, memory, whether loaded
    POST /v1/complete              → multipart: images (1–4 RGBA PNG crops; alpha = object
                                     mask), params (JSON) → model/gltf-binary (+Y up)

Each image is one *view* of the same object, largest first. Alpha is honoured directly,
so no background removal model runs. Models load on the first /v1/complete, not on
/v1/health, so a local worker costs no memory until it is used.

Run:  VERONICA_COMPLETER_TOKEN=secret uvicorn worker:app --host 0.0.0.0 --port 8790
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import platform
import subprocess
import tempfile
import threading
import time
from typing import Protocol

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
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
    # hunyuan3d-mini
    steps: int | None = Field(None, ge=1, le=100)  # None = the checkpoint's default
    guidance: float = Field(5.0, ge=0, le=15)
    octree_resolution: int = Field(256, ge=128, le=512)
    max_faces: int = Field(60000, ge=2000, le=400000)


class Engine(Protocol):
    name: str

    def info(self) -> dict: ...

    def generate(self, images: list[Image.Image], p: Params) -> bytes: ...


class TrellisEngine:
    """The real thing. Imported lazily so the HTTP layer can be tested without CUDA."""

    @classmethod
    def describe(cls) -> dict:
        return {"engine": "trellis", "model": os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-image-large"), "license": "MIT", "device": "cuda"}

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


def _torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _device_info(device: str) -> dict:
    if device == "cuda":
        import torch

        props = torch.cuda.get_device_properties(0)
        return {"device": props.name, "vram_gb": round(props.total_memory / 2**30, 1)}
    if platform.system() == "Darwin":
        def sysctl(k: str) -> str:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()

        chip = sysctl("machdep.cpu.brand_string") or "Apple Silicon"
        mem = int(sysctl("hw.memsize") or 0)
        return {"device": f"{chip} GPU (Metal)" if device == "mps" else f"{chip} CPU", "vram_gb": round(mem / 2**30, 1), "unified_memory": True}
    return {"device": platform.processor() or "cpu", "vram_gb": 0}


class HunyuanMiniEngine:
    """Hunyuan3D-2mini (0.6B shape DiT + FlashVDM decoder), pure PyTorch: no compiled CUDA
    extensions, so this is the engine that runs on a Mac. It produces an untextured mesh;
    the backend colours it from the photo. It is conditioned on one image, so with several
    views the first (the backend sends the largest mask first) is used."""

    REPO = "tencent/Hunyuan3D-2mini"
    LICENSE = "Tencent Hunyuan 3D 2.0 Community License (not licensed in the EU, UK or South Korea)"
    DEFAULT_STEPS = {"turbo": 5, "fast": 30, "": 30}

    @classmethod
    def describe(cls) -> dict:
        sub = os.environ.get("HUNYUAN_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo")
        return {"engine": "hunyuan3d-mini", "model": f"{cls.REPO}/{sub}", "license": cls.LICENSE, **_device_info(_torch_device())}

    def __init__(self) -> None:
        import torch
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline

        self.torch = torch
        self.sub = os.environ.get("HUNYUAN_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo")
        self.name = f"{self.REPO}/{self.sub}"
        self.device = _torch_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        # hy3dgen's from_pretrained downloads the whole subfolder, i.e. a .ckpt duplicate of
        # the weights (another 3.8 GB); fetch only the config and the safetensors.
        from huggingface_hub import snapshot_download

        files = [f"{self.sub}/config.yaml", f"{self.sub}/model.fp16.safetensors"]
        root = snapshot_download(self.REPO, allow_patterns=files + ["LICENSE", "NOTICE"])
        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            os.path.join(root, files[1]), os.path.join(root, files[0]), device=self.device, dtype=dtype, use_safetensors=True,
            from_pretrained_kwargs={"model_path": self.REPO, "subfolder": self.sub, "use_safetensors": True, "variant": "fp16", "dtype": dtype, "device": self.device})
        for attr in ("sigmas", "timesteps"):  # schedulers build these in float64, which MPS cannot hold
            v = getattr(self.pipe.scheduler, attr, None)
            if torch.is_tensor(v) and v.dtype == torch.float64:
                setattr(self.pipe.scheduler, attr, v.float())
        self.variant = next((v for v in ("turbo", "fast") if self.sub.endswith(v)), "")
        if self.variant == "turbo" and self.device == "cuda":
            # FlashVDM's adaptive decoder finds no surface on MPS (verified on an M5), so Macs
            # keep the checkpoint's own hierarchical decoder, which is correct there.
            self.pipe.enable_flashvdm(mc_algo="mc")
        self.post = (FloaterRemover(), DegenerateFaceRemover(), FaceReducer())

    def info(self) -> dict:
        return {"engine": "hunyuan3d-mini", "model": self.name, "license": self.LICENSE, **_device_info(self.device)}

    def generate(self, images: list[Image.Image], p: Params) -> bytes:
        from hy3dgen.shapegen.pipelines import export_to_trimesh

        gen = self.torch.Generator("cpu").manual_seed(p.seed)  # CPU noise: same seed, same shape on every device
        latents = self.pipe(image=images[0], num_inference_steps=p.steps or self.DEFAULT_STEPS[self.variant], guidance_scale=p.guidance,
                            generator=gen, output_type="latent", enable_pbar=False)
        with self.torch.no_grad():
            vae = self.pipe.vae
            grid = self.decode_volume(vae(latents / vae.scale_factor), p.octree_resolution)
            out = vae.surface_extractor(grid[None], mc_level=0.0, bounds=self.BOUNDS, octree_resolution=p.octree_resolution)
        mesh = export_to_trimesh(out)[0]
        if mesh is None or len(mesh.faces) == 0:
            raise RuntimeError("the model produced no surface for this crop")
        floaters, degenerate, reducer = self.post
        mesh = degenerate(floaters(mesh))
        if len(mesh.faces) > p.max_faces:
            try:  # fast-simplification: ~1 s for a million faces, vs ~100 s in pymeshlab
                mesh = mesh.simplify_quadric_decimation(face_count=p.max_faces)
            except ImportError:
                mesh = reducer(mesh, max_facenum=p.max_faces)
        data = mesh.export(file_type="glb")
        if self.device == "mps":
            self.torch.mps.empty_cache()
        elif self.device == "cuda":
            self.torch.cuda.empty_cache()
        return data

    BOUNDS = 1.01

    def decode_volume(self, z, res: int, coarse: int = 64, chunk: int = 20000):
        """Occupancy logits on a (res+1)^3 grid, evaluated densely only near the surface.

        hy3dgen's hierarchical and FlashVDM decoders find no surface on MPS, and the dense
        decoder spends ~70 s on 17M queries at res 256. This decodes a coarse grid, marks
        the coarse cells whose corners change sign (plus one cell of margin), queries the
        fine grid only inside them, and fills everything else from the upsampled coarse
        grid, whose sign is already right there. Same surface, ~10× fewer queries."""
        torch, F = self.torch, self.torch.nn.functional
        dev, b = z.device, self.BOUNDS
        geo = self.pipe.vae.geo_decoder

        def query(pts):
            out = [geo(queries=pts[i:i + chunk][None].to(z.dtype), latents=z).float().reshape(-1) for i in range(0, len(pts), chunk)]
            return torch.cat(out)

        coarse = min(coarse, res)
        ax = torch.linspace(-b, b, coarse + 1, device=dev)
        c = query(torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)).reshape((1, 1) + (coarse + 1,) * 3)
        hi, lo = F.max_pool3d(c, 2, 1), -F.max_pool3d(-c, 2, 1)
        active = F.max_pool3d(((hi > 0) & (lo < 0)).float(), 3, 1, padding=1)  # crossing cells + 1-cell margin
        fine = F.interpolate(c, size=(res + 1,) * 3, mode="trilinear", align_corners=True)[0, 0]
        k = res // coarse
        cells = active[0, 0].repeat_interleave(k, 0).repeat_interleave(k, 1).repeat_interleave(k, 2)
        if cells.shape[0] != res:  # res not a multiple of coarse
            cells = F.interpolate(active, size=(res,) * 3, mode="nearest")[0, 0]
        verts = F.max_pool3d(F.pad(cells[None, None], (1, 1, 1, 1, 1, 1)), 2, 1)[0, 0] > 0  # vertices of active cells
        idx = verts.nonzero()
        fine[verts] = query(idx.float() * (2 * b / res) - b)
        return fine

    def close(self) -> None:
        del self.pipe
        import gc

        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()


ENGINES = {"trellis": TrellisEngine, "hunyuan3d-mini": HunyuanMiniEngine}


def pick_engine(name: str | None = None) -> type:
    """VERONICA_COMPLETER_ENGINE, or (auto) whichever engine's package is installed."""
    name = (name or os.environ.get("VERONICA_COMPLETER_ENGINE", "") or "auto").strip().lower()
    if name in ENGINES:
        return ENGINES[name]
    if name != "auto":
        raise ValueError(f"unknown VERONICA_COMPLETER_ENGINE {name!r}; use one of {', '.join(ENGINES)} or auto")
    if importlib.util.find_spec("trellis") is None and importlib.util.find_spec("hy3dgen") is not None:
        return HunyuanMiniEngine
    return TrellisEngine


def create_app(engine_factory=None, idle_unload_s: float | None = None) -> FastAPI:
    """`idle_unload_s` (or VERONICA_COMPLETER_IDLE_UNLOAD) frees the model after that many idle
    seconds; useful when the worker shares unified memory with the rest of VERONICA."""
    engine_factory = engine_factory or pick_engine()
    if idle_unload_s is None:
        idle_unload_s = float(os.environ.get("VERONICA_COMPLETER_IDLE_UNLOAD", "0") or 0)
    app = FastAPI(title="VERONICA completer worker", version=API_VERSION)
    state: dict = {"engine": None, "error": None, "last_used": 0.0}
    lock = threading.Lock()  # one generation at a time per GPU
    token = os.environ.get("VERONICA_COMPLETER_TOKEN")
    describe = getattr(engine_factory, "describe", None)

    def load_locked() -> Engine:
        """Call with `lock` held."""
        if state["engine"] is None:
            try:
                state["engine"] = engine_factory()
                state["error"] = None
            except Exception as e:  # surfaced through /v1/health
                state["error"] = f"{type(e).__name__}: {e}"
                raise HTTPException(503, f"model failed to load: {state['error']}")
        return state["engine"]

    def engine() -> Engine:
        with lock:
            return load_locked()

    def run_generation(ims: list[Image.Image], p: Params) -> tuple[Engine, bytes]:
        with lock:  # the idle unloader takes the same lock, so the model cannot vanish mid-generation
            eng = load_locked()
            try:
                return eng, eng.generate(ims, p)
            except Exception as e:
                raise HTTPException(500, f"generation failed: {type(e).__name__}: {e}")
            finally:
                state["last_used"] = time.time()

    def unload_when_idle() -> None:
        while True:
            time.sleep(max(1.0, idle_unload_s / 4))
            if state["engine"] is not None and time.time() - state["last_used"] > idle_unload_s and lock.acquire(blocking=False):
                try:
                    eng, state["engine"] = state["engine"], None
                    if hasattr(eng, "close"):
                        eng.close()
                finally:
                    lock.release()

    if idle_unload_s > 0:
        threading.Thread(target=unload_when_idle, daemon=True).start()

    def auth(authorization: str | None = Header(None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong bearer token")

    @app.get("/v1/health")
    def health(_: None = Depends(auth)) -> dict:
        if state["engine"] is not None:
            return {"ok": True, "api": API_VERSION, "loaded": True, **state["engine"].info()}
        if describe is not None:  # report without loading: the model loads on first use
            try:
                info = describe()
            except Exception as e:
                return {"ok": False, "api": API_VERSION, "error": f"{type(e).__name__}: {e}"}
            if state["error"]:
                return {"ok": False, "api": API_VERSION, "loaded": False, "error": state["error"], **info}
            return {"ok": True, "api": API_VERSION, "loaded": False, **info}
        try:
            info = engine().info()
            return {"ok": True, "api": API_VERSION, "loaded": True, **info}
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
        t0 = time.perf_counter()
        eng, glb = await run_in_threadpool(run_generation, ims, p)  # keeps /v1/health answering meanwhile
        meta = {"api": API_VERSION, "model": eng.name, "seconds": round(time.perf_counter() - t0, 2), "views": len(ims), "params": p.model_dump()}
        return Response(glb, media_type="model/gltf-binary", headers={"X-Veronica-Meta": json.dumps(meta)})

    return app


app = create_app()
