"""Generative completion: worker HTTP contract + VERONICA-side alignment/carving/persistence.

TRELLIS itself needs CUDA, so the worker's model engine is replaced by a test double that
returns a known shape (the synthetic room's cube at the wrong scale, heading and place,
with a texture). Everything else — HTTP, GLB decoding, alignment, carving, manifest
updates, persistence across rebuilds, revert — is the production code path."""
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from veronica.assets import read_packet
from veronica.pipeline.build import build_scene
from veronica.pipeline.generate import generate_completion, revert_generated
from veronica.providers.generative import CompleterClient

WORKER = Path(__file__).resolve().parents[2] / "completer-worker" / "worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("veronica_worker", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _textured_cube_glb(scale=3.0, yaw_deg=30.0, offset=(4, 2, -3)) -> bytes:
    box = trimesh.creation.box((0.5, 0.5, 0.5)).subdivide().subdivide()
    tex = np.zeros((64, 64, 3), np.uint8)
    tex[:32] = (220, 40, 40)  # top half red, bottom half blue: checks the UV convention
    tex[32:] = (40, 40, 220)
    uv = np.c_[(box.vertices[:, 0] + 0.25) * 2, (box.vertices[:, 1] + 0.25) * 2]
    box.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(tex))
    box.apply_scale(scale)
    box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw_deg), (0, 1, 0)))
    box.apply_translation(offset)
    return box.export(file_type="glb")


class CubeEngine:
    name = "test-double/cube"

    def __init__(self):
        self.calls = []

    def info(self):
        return {"model": self.name, "device": "cpu", "vram_gb": 0}

    def generate(self, images, p):
        self.calls.append((len(images), [im.mode for im in images], p))
        return _textured_cube_glb()


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("VERONICA_COMPLETER_TOKEN", "s3cret")
    mod = _load_worker()
    eng = CubeEngine()
    app = mod.create_app(lambda: eng)
    return TestClient(app), eng


def _png(mode="RGBA"):
    b = io.BytesIO()
    Image.new(mode, (32, 32)).save(b, format="PNG")
    return b.getvalue()


def test_worker_contract(worker):
    client, eng = worker
    assert client.get("/v1/health").status_code == 401
    h = client.get("/v1/health", headers={"Authorization": "Bearer s3cret"}).json()
    assert h["ok"] and h["api"] == "veronica-completer/1"
    auth = {"Authorization": "Bearer s3cret"}
    r = client.post("/v1/complete", files=[("images", ("a.png", _png("RGB"), "image/png"))], headers=auth)
    assert r.status_code == 422  # alpha mask required
    r = client.post("/v1/complete", files=[("images", ("a.png", _png(), "image/png"))], data={"params": '{"ss_steps": 999}'}, headers=auth)
    assert r.status_code == 422  # params validated
    r = client.post("/v1/complete", files=[("images", ("a.png", _png(), "image/png")), ("images", ("b.png", _png(), "image/png"))], headers=auth)
    assert r.status_code == 200 and r.headers["content-type"] == "model/gltf-binary"
    assert json.loads(r.headers["X-Veronica-Meta"])["views"] == 2 and eng.calls[-1][0] == 2


def test_generated_completion_end_to_end(room, tmp_path, worker):
    http, eng = worker
    man = build_scene(room, tmp_path, "gen")
    room.save(tmp_path / "analysis.npz")
    box = next(n for n in man["nodes"] if n["id"] == "obj-0")
    assert box["provenance"]["completion"] == "mirror-symmetry"
    client = CompleterClient("http://testserver", token="s3cret", client=http)

    meta = generate_completion(tmp_path, "obj-0", client, {"seed": 3})
    assert meta["accepted"], meta["alignment"]
    assert eng.calls[-1][1] == ["RGBA"]  # crop with mask in alpha
    al = meta["alignment"]
    assert abs(al["scale"] * 3.0 - 1.0) < 0.08  # generated cube (×3) brought back to 0.5 m
    assert al["silhouette_iou"] > 0.6

    man = json.loads((tmp_path / "manifest.json").read_text())
    node = next(n for n in man["nodes"] if n["id"] == "obj-0")
    assert node["provenance"]["completion"] == "generative"
    assert node["provenance"]["prior"]["completion"] == "mirror-symmetry"
    assert man["textures"]["gen-obj-0"] == "gen-obj-0.png"
    comp = read_packet(tmp_path / node["representations"]["completion"]["file"])
    world = comp["position"] + np.array(node["transform"]["position"])
    # only the unseen part survives: the back of the cube, behind the observed front face
    cam_z = np.array(man["camera"]["worldFromCamera"])[2, 3]
    assert (comp["observed"] == 0).all()
    assert np.median(cam_z - world[:, 2]) > 2.6  # front face is 2.5 m from the camera
    assert "uv" in comp and comp["uv"].min() >= -1e-3 and comp["uv"].max() <= 1 + 1e-3
    # cube dimensions in the manifest stay metric
    assert max(abs(d - 0.5) for d in node["dimensions"]) < 0.08

    # survives a rebuild of the scene
    man2 = build_scene(room, tmp_path, "gen")
    assert next(n for n in man2["nodes"] if n["id"] == "obj-0")["provenance"]["completion"] == "generative"

    # and can be reverted to the symmetry prior
    man3 = revert_generated(tmp_path, "obj-0")
    n3 = next(n for n in man3["nodes"] if n["id"] == "obj-0")
    assert n3["provenance"]["completion"] == "mirror-symmetry" and "gen-obj-0" not in man3["textures"]
    assert not (tmp_path / "gen-obj-0.vrnc").exists()


def test_rejects_shapes_that_do_not_fit(room, tmp_path, worker, monkeypatch):
    http, eng = worker
    build_scene(room, tmp_path, "gen")
    room.save(tmp_path / "analysis.npz")
    pole = trimesh.creation.cylinder(radius=0.02, height=2.0)
    monkeypatch.setattr(eng, "generate", lambda images, p: pole.export(file_type="glb"))
    with pytest.raises(RuntimeError, match="rejected"):
        generate_completion(tmp_path, "obj-0", CompleterClient("http://testserver", token="s3cret", client=http))
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert next(n for n in man["nodes"] if n["id"] == "obj-0")["provenance"]["completion"] == "mirror-symmetry"
