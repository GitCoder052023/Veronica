import json

import numpy as np

from veronica.assets import read_packet
from veronica.pipeline.build import build_scene


def test_build_scene_on_synthetic_room(room, tmp_path):
    man = build_scene(room, tmp_path, "test", source_name="room.png")
    kinds = {n["id"]: n for n in man["nodes"]}
    assert "obj-0" in kinds and "structure-floor" in kinds
    assert man["world"]["upSource"] == "floor"
    box = kinds["obj-0"]
    w, h, d = box["dimensions"]
    # metric recovery of a 0.5 m cube from exact depth (completion supplies the hidden half)
    assert abs(w - 0.5) < 0.05 and abs(h - 0.5) < 0.05 and 0.35 < d < 0.65
    # the box rests on the floor
    pos, lo = np.array(box["transform"]["position"]), np.array(box["bounds"]["min"])
    assert abs((pos + lo)[1]) < 0.04
    assert box["relations"]["supportedBy"] == "structure-floor"
    assert 0 < box["provenance"]["observedFraction"] < 1
    # assets exist, are finite and carry provenance flags
    surf = read_packet(tmp_path / box["representations"]["surface"]["file"])
    assert np.isfinite(surf["position"]).all() and (surf["observed"] == 255).all()
    comp = read_packet(tmp_path / box["representations"]["completion"]["file"])
    assert (comp["observed"] == 0).all()
    # floor continues behind the box and that part is flagged inferred
    floor = read_packet(tmp_path / kinds["structure-floor"]["representations"]["surface"]["file"])
    assert (floor["observed"] == 0).any() and abs(floor["position"][:, 1]).max() < 1e-3
    assert json.loads((tmp_path / "manifest.json").read_text())["id"] == "test"


def test_camera_pose_reprojects_the_photo(room, tmp_path):
    man = build_scene(room, tmp_path, "t2")
    M = np.array(man["camera"]["worldFromCamera"])
    # camera height above the floor is recovered
    assert abs(M[1, 3] - 1.2) < 0.03
    # GL camera looks down -Z of its frame: forward is horizontal (level camera)
    fwd = -M[:3, 2]
    assert abs(fwd[1]) < 0.02
