"""HTTP API. Run: `uv run uvicorn veronica.api:app --port 8787`."""
from __future__ import annotations

import io
import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import MODELS, settings
from .jobs import JobManager
from .pipeline.analysis import analyze, load_image
from .pipeline.build import build_scene
from .pipeline.generate import generate_completion, revert_generated
from .pipeline.refine import add_objects
from .providers import registry
from .providers.generative import CompleterClient

app = FastAPI(title="VERONICA — The Spatial Intelligence System", version="0.1.0", license_info={"name": "Apache-2.0"})
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs = JobManager()
completer = CompleterClient()
_completer_status: dict = {"t": 0.0, "v": None}


def completer_status() -> dict:
    """Worker health, cached briefly so /api/health stays fast."""
    import time

    if _completer_status["v"] is None or time.time() - _completer_status["t"] > 15:
        _completer_status.update(t=time.time(), v=completer.health())
    return _completer_status["v"]
MAX_UPLOAD = 40 * 1024 * 1024
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tif", ".tiff"}


def _scene_dir(scene_id: str) -> Path:
    if not scene_id.replace("-", "").isalnum():
        raise HTTPException(400, "bad scene id")
    d = settings.scenes_dir / scene_id
    if not (d / "manifest.json").exists():
        raise HTTPException(404, "scene not found")
    return d


@app.get("/api/health")
def health() -> dict:
    from .pipeline.multiview import multiview_status

    return {
        "ok": True, "device": settings.device, "loaded": registry.loaded(),
        "depthModel": settings.depth_model,
        "models": {k: {"role": m.role, "id": m.hf_id, "license": m.license, "note": m.note} for k, m in MODELS.items()},
        "multiView": multiview_status(),
        "completer": completer_status(),
    }


async def _save_uploads(d: Path, files: list[UploadFile], start: int = 0) -> list[Path]:
    paths = []
    for i, f in enumerate(files, start):
        ext = Path(f.filename or "").suffix.lower() or ".jpg"
        if ext not in ALLOWED:
            raise HTTPException(415, f"unsupported file type {ext}")
        data = await f.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "file too large")
        p = d / "inputs" / f"{i:03d}{ext}"
        p.write_bytes(data)
        paths.append(p)
    return paths


def _submit_reconstruction(d: Path, sid: str, paths: list[Path], name: str, opts: dict) -> dict:
    vocab = [v.strip() for v in opts.get("vocabulary", []) if v.strip()] or None
    depth_model = opts.get("depthModel") or None
    if len(paths) == 1:
        def run(progress):
            progress("load", 0.01, "Decoding photo")
            img, f = load_image(paths[0])
            an = analyze(img, f, vocab, progress, depth_model)
            an.save(d / "analysis.npz")
            build_scene(an, d, sid, progress, name)
            return sid
        kind = "single-view"
    else:
        from .pipeline.multiview import reconstruct_multiview

        def run(progress):
            reconstruct_multiview(paths, d, sid, progress, vocab, name)
            return sid
        kind = "multi-view"
    return {"job": jobs.submit(kind, run, sid).public()}


@app.post("/api/scenes")
async def create_scene(files: list[UploadFile] = File(...), options: str = Form("{}")) -> dict:
    sid = uuid.uuid4().hex[:10]
    d = settings.scenes_dir / sid
    (d / "inputs").mkdir(parents=True)
    try:
        paths = await _save_uploads(d, files)
    except HTTPException:
        shutil.rmtree(d)
        raise
    return _submit_reconstruction(d, sid, paths, files[0].filename or "photo", json.loads(options or "{}"))


def _scene_photos(d: Path, man: dict) -> list[Path]:
    """The original photos a scene was built from (uploads, or the saved working images)."""
    ups = sorted(p for p in (d / "inputs").glob("*") if p.suffix.lower() in ALLOWED) if (d / "inputs").exists() else []
    if ups:
        return ups
    if man["mode"] == "multi-view":
        return [d / c["image"] for c in man.get("cameras", []) if (d / c["image"]).exists()]
    return [d / man["textures"]["image"]]


@app.post("/api/scenes/{scene_id}/extend")
async def extend_scene(scene_id: str, files: list[UploadFile] = File(...), options: str = Form("{}")) -> dict:
    """Rebuild a scene as multi-view from its photo(s) plus new photos taken from other
    angles — the way to recover real hidden sides without a generative GPU worker."""
    src = _scene_dir(scene_id)
    man = json.loads((src / "manifest.json").read_text())
    old = _scene_photos(src, man)
    if len(old) + len(files) < 2:
        raise HTTPException(422, "add at least one more photo")
    from .pipeline.multiview import MAX_VIEWS

    if len(old) + len(files) > MAX_VIEWS:
        raise HTTPException(422, f"at most {MAX_VIEWS} photos per scene ({len(old)} already used)")
    sid = uuid.uuid4().hex[:10]
    d = settings.scenes_dir / sid
    (d / "inputs").mkdir(parents=True)
    paths = []
    for i, p in enumerate(old):
        q = d / "inputs" / f"{i:03d}{p.suffix.lower()}"
        shutil.copyfile(p, q)
        paths.append(q)
    try:
        paths += await _save_uploads(d, files, len(paths))
    except HTTPException:
        shutil.rmtree(d)
        raise
    base = (man["source"].get("name") or "scene").rsplit(" + ", 1)[0]
    return _submit_reconstruction(d, sid, paths, f"{base} + {len(paths) - 1} angles", json.loads(options or "{}"))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j.public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    return {"cancelled": jobs.cancel(job_id)}


@app.get("/api/scenes")
def list_scenes() -> list[dict]:
    out = []
    for m in sorted(settings.scenes_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            man = json.loads(m.read_text())
            out.append({"id": man["id"], "created": man["created"], "name": man["source"].get("name"), "mode": man["mode"],
                        "objects": sum(1 for n in man["nodes"] if n["kind"] == "object")})
        except Exception:
            continue
    return out


@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str) -> dict:
    return json.loads((_scene_dir(scene_id) / "manifest.json").read_text())


@app.get("/api/scenes/{scene_id}/files/{name}")
def get_file(scene_id: str, name: str):
    d = _scene_dir(scene_id)
    p = (d / name).resolve()
    if p.parent != d.resolve() or not p.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(p, headers={"Cache-Control": "no-cache"})


@app.delete("/api/scenes/{scene_id}")
def delete_scene(scene_id: str) -> dict:
    shutil.rmtree(_scene_dir(scene_id))
    return {"deleted": scene_id}


class AddObjectRequest(BaseModel):
    points: list[tuple[float, float]] | None = Field(None, description="pixel coords in the working image")
    pointLabels: list[int] | None = None
    text: str | None = None
    label: str | None = None


@app.post("/api/scenes/{scene_id}/objects")
def add_object(scene_id: str, req: AddObjectRequest) -> dict:
    d = _scene_dir(scene_id)
    if not (d / "analysis.npz").exists():
        raise HTTPException(409, "scene has no single-view analysis to refine")
    if not req.points and not req.text:
        raise HTTPException(422, "give points or text")

    def run(progress):
        add_objects(d, points=req.points, point_labels=req.pointLabels, text=req.text, label=req.label, progress=progress)
        return scene_id

    return {"job": jobs.submit("refine", run, scene_id).public()}


@app.post("/api/models/unload")
def unload_models() -> dict:
    registry.unload_all()
    return {"loaded": registry.loaded()}


class GenerateRequest(BaseModel):
    ids: list[str] | None = Field(None, description="objects to complete; default = every non-flat object")
    seed: int = 1
    steps: int = Field(12, ge=4, le=50)


def _completer_or_409():
    st = completer_status()
    if not st.get("configured"):
        raise HTTPException(409, "No generative completer configured. Run completer-worker on a CUDA machine and set VERONICA_COMPLETER_URL.")
    if not st.get("ok"):
        _completer_status["v"] = None  # retry fresh next time
        raise HTTPException(503, f"Generative completer unreachable: {st.get('reason') or st.get('error')}")


@app.post("/api/scenes/{scene_id}/generate")
def generate(scene_id: str, req: GenerateRequest) -> dict:
    d = _scene_dir(scene_id)
    _completer_or_409()
    man = json.loads((d / "manifest.json").read_text())
    ids = req.ids or [n["id"] for n in man["nodes"] if n["kind"] == "object" and not n.get("flat")]
    params = {"seed": req.seed, "ss_steps": req.steps, "slat_steps": req.steps}

    def run(progress):
        done, failed = [], []
        for i, oid in enumerate(ids):
            sub = lambda stage, f, msg="": progress(stage, (i + f) / len(ids), f"[{i + 1}/{len(ids)}] {msg}")
            try:
                generate_completion(d, oid, completer, params, sub)
                done.append(oid)
            except Exception as e:  # one bad object should not stop the batch
                failed.append(f"{oid}: {e}")
                progress("generate", (i + 1) / len(ids), f"{oid} kept its symmetry prior — {e}")
        if not done:
            raise RuntimeError("; ".join(failed) or "nothing generated")
        if failed:
            progress("done", 1.0, f"Generated {len(done)}, kept prior for {len(failed)}: " + "; ".join(failed))
        return scene_id

    return {"job": jobs.submit("generate", run, scene_id).public()}


@app.delete("/api/scenes/{scene_id}/objects/{oid}/generate")
def revert_generate(scene_id: str, oid: str) -> dict:
    d = _scene_dir(scene_id)
    revert_generated(d, oid)
    return {"reverted": oid}
