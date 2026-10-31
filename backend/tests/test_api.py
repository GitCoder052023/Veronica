from fastapi.testclient import TestClient

from veronica.api import app

client = TestClient(app)


def test_health_lists_models_and_licenses():
    r = client.get("/api/health").json()
    assert r["ok"] and "depth-pro" in r["models"]
    assert all(m["license"] for m in r["models"].values())


def test_missing_scene_and_bad_ids():
    assert client.get("/api/scenes/doesnotexist").status_code == 404
    assert client.get("/api/scenes/..%2F..%2Fetc/files/passwd").status_code in (400, 404)
    assert client.get("/api/jobs/nope").status_code == 404


def test_rejects_non_images():
    r = client.post("/api/scenes", files={"files": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415


def test_extend_validates_and_finds_original_photos(room, tmp_path, monkeypatch):
    import io
    import json

    from PIL import Image

    from veronica import api
    from veronica.config import settings
    from veronica.pipeline.build import build_scene

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    d = settings.scenes_dir / "abc123"
    man = build_scene(room, d, "abc123", source_name="room.png")
    # a CLI-built scene has no inputs/ — its working image is the original
    assert api._scene_photos(d, man) == [d / "image.jpg"]
    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    many = [("files", (f"{i}.png", png.getvalue(), "image/png")) for i in range(12)]
    r = client.post("/api/scenes/abc123/extend", files=many)
    assert r.status_code == 422 and "at most" in r.json()["detail"]
    r = client.post("/api/scenes/abc123/extend", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 415
    assert not any(p.name != "abc123" for p in settings.scenes_dir.iterdir())  # failed uploads leave nothing behind
    assert json.loads((d / "manifest.json").read_text())["id"] == "abc123"


def test_views_with_other_aspect_are_cropped_not_distorted():
    from PIL import Image

    from veronica.pipeline.multiview import _match_frame

    portrait = Image.new("RGB", (300, 400), "red")
    out = _match_frame(portrait, 400, 300)
    assert out.size == (400, 300)


def test_heic_is_readable():
    import pillow_heif  # noqa: F401  (registered by the analysis module)
    from PIL import Image

    import veronica.pipeline.analysis  # noqa: F401

    assert ".heic" in Image.registered_extensions()
