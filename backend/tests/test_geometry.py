import numpy as np

from veronica.geometry.camera import Intrinsics, depth_discontinuities
from veronica.geometry.completion import MirrorCompleter, ObjectContext, RevolveCompleter
from veronica.geometry.mesh import Mesh, grid_mesh
from veronica.geometry.planes import classify_planes, ransac_planes, world_frame
from veronica.pipeline.build import carve_free_space, resolve_instances


def test_unproject_project_roundtrip():
    K = Intrinsics(500, 500, 160, 120, 320, 240)
    d = np.full((240, 320), 2.0, np.float32)
    P = K.unproject(d).reshape(-1, 3)
    u, v, z = K.project(P)
    assert np.allclose(z, 2.0)
    assert np.allclose(u.reshape(240, 320)[0], np.arange(320))


def test_grid_mesh_cuts_occlusion_edges():
    K = Intrinsics(300, 300, 50, 50, 100, 100)
    d = np.full((100, 100), 2.0, np.float32)
    d[:, 50:] = 4.0  # step: two surfaces, must not be bridged
    m = grid_mesh(K.unproject(d), np.ones((100, 100), bool))
    z = m.positions[m.indices][..., 2]
    assert (z.max(1) / z.min(1)).max() < 1.1
    assert depth_discontinuities(d)[:, 49:51].all()


def test_ransac_finds_floor_and_frame_is_orthonormal(room):
    from veronica.geometry.camera import Intrinsics

    K = Intrinsics(room.fx, room.fy, room.cx, room.cy, room.depth.shape[1], room.depth.shape[0])
    P = K.unproject(room.depth).reshape(-1, 3)
    P = P[np.isfinite(P).all(1) & ~room.masks[0].ravel()]
    planes = ransac_planes(P[::7])
    up = classify_planes(planes, P[::7])
    floor = next(p for p in planes if p.kind == "floor")
    assert np.allclose(floor.normal, [0, -1, 0], atol=0.02)
    assert abs(floor.d - 1.2) < 0.02
    fr = world_frame(up, floor, np.array([0, 0.5, 2.5]), [p for p in planes if p.kind == "wall"])
    assert np.allclose(fr.R @ fr.R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(fr.R), 1.0)
    # a floor point maps to y = 0, the camera to y = 1.2
    assert abs(fr.apply(np.array([[0.3, 1.2, 3.0]]))[0, 1]) < 1e-3
    assert abs(fr.t[1] - 1.2) < 0.02


def test_resolve_instances_small_wins_overlap():
    big = np.zeros((50, 50), bool)
    big[5:45, 5:45] = True
    small = np.zeros((50, 50), bool)
    small[20:30, 20:30] = True
    labels, kept = resolve_instances(np.stack([big, small]), [0.9, 0.9], min_area_frac=0)
    assert (labels[25, 25], labels[10, 10]) == (1, 0)
    assert kept == [0, 1]


def _box_front_face(w=0.5, h=0.5, z0=-2.0):
    """Front face of a box in a Y-up world, camera at origin looking down -Z."""
    xs, ys = np.meshgrid(np.linspace(-w / 2, w / 2, 30), np.linspace(0, h, 30))
    P = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z0)], 1).astype(np.float32)
    idx = np.arange(900).reshape(30, 30)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]).astype(np.uint32)
    return Mesh(P, tris, observed=np.full(900, 255, np.uint8))


def test_mirror_completion_uses_prior_when_side_unseen():
    S = _box_front_face()
    c = MirrorCompleter().complete(ObjectContext("box", S, np.array([0.0, 0.25, 0.0]), 0.0, 0.02))
    assert c.mesh is not None and (c.mesh.observed == 0).all()
    zs = c.mesh.positions[:, 2]
    # prior depth for "box" is 1.0 × width → back face 0.5 m behind the front
    assert np.isclose(zs.min(), -2.5, atol=0.03)
    assert c.params["depth_source"].startswith("class prior")


def test_revolve_completion_recovers_cylinder_radius():
    r, cz = 0.04, -1.0
    ang = np.linspace(np.pi * 0.15, np.pi * 0.85, 40)  # front arc only, facing +Z (camera)
    ys = np.linspace(0, 0.25, 30)
    A, Y = np.meshgrid(ang, ys)
    P = np.stack([r * np.cos(A).ravel(), Y.ravel(), cz + r * np.sin(A).ravel()], 1).astype(np.float32)
    idx = np.arange(P.shape[0]).reshape(30, 40)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]).astype(np.uint32)
    S = Mesh(P, tris, uvs=np.zeros((len(P), 2), np.float32), observed=np.full(len(P), 255, np.uint8))
    comp = RevolveCompleter().complete(ObjectContext("bottle", S, np.array([0.0, 0.1, 0.5]), 0.0, 0.003))
    assert comp.method == "surface-of-revolution"
    assert abs(comp.params["mean_radius_m"] - r) < 0.006
    assert np.allclose(comp.params["axis_xz"], [0.0, cz], atol=0.01)


def test_carving_removes_geometry_in_front_of_observation():
    K = Intrinsics(100, 100, 50, 50, 100, 100)
    depth = np.full((100, 100), 2.0, np.float32)
    R, t = np.eye(3), np.zeros(3)
    P = np.array([[0, 0, 1.0], [0.01, 0, 1.0], [0, 0.01, 1.0], [0, 0, 3.0], [0.01, 0, 3.0], [0, 0.01, 3.0]], np.float32)
    m = Mesh(P, np.array([[0, 1, 2], [3, 4, 5]], np.uint32), observed=np.zeros(6, np.uint8))
    carved, frac = carve_free_space(m, R, t, K, depth)
    assert frac == 0.5 and carved.n_faces == 1 and carved.positions[:, 2].min() > 2.5
