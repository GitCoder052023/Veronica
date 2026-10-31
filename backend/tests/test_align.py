import numpy as np
import trimesh

from veronica.geometry.align import ViewSilhouette, align_generated
from veronica.geometry.camera import Intrinsics


def _l_shape() -> trimesh.Trimesh:
    """Asymmetric object (an 'L' of two boxes) so heading is identifiable."""
    a = trimesh.creation.box((0.6, 0.4, 0.3))
    b = trimesh.creation.box((0.2, 0.6, 0.3))
    b.apply_translation((0.2, 0.5, 0.0))
    return trimesh.util.concatenate([a, b])


def _scene(yaw_deg=37.0, scale=0.8, t=(0.3, 0.0, -2.5)):
    canon = _l_shape()
    canon.apply_translation(-canon.bounds.mean(0))
    true = canon.copy()
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    M = np.eye(4)
    M[:3, :3] = R * scale
    true.apply_transform(M)
    true.apply_translation(np.array(t) - np.array([0, true.bounds[0, 1], 0]))  # stand on y = 0
    # camera 1.2 m up, looking at the object along -Z (OpenCV camera: x right, y down, z forward)
    cam = np.array([0.0, 1.2, 0.0])
    look = true.bounds.mean(0) - cam
    fwd = look / np.linalg.norm(look)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_cw = np.stack([right, down, fwd])
    t_cw = -R_cw @ cam
    K = Intrinsics(400, 400, 160, 120, 320, 240)
    # observed = surface samples that win a z-buffer test from the camera (visible surface)
    pts, _ = trimesh.sample.sample_surface_even(true, 60000, seed=1)
    pc = pts @ R_cw.T + t_cw
    u, v, z = K.project(pc)
    ui, vi = u.astype(int), v.astype(int)
    ok = (ui >= 0) & (ui < 320) & (vi >= 0) & (vi < 240) & (z > 0)
    zbuf = np.full((240, 320), np.inf)
    np.minimum.at(zbuf, (vi[ok], ui[ok]), z[ok])
    vis = np.zeros(len(pts), bool)
    vis[ok] = z[ok] <= zbuf[vi[ok], ui[ok]] + 0.01
    obs = pts[vis]
    # silhouette mask from the full object
    allp = true.sample(40000, return_index=False) @ R_cw.T + t_cw
    u, v, z = K.project(allp)
    mask = np.zeros((240, 320), bool)
    ok = (u >= 0) & (u < 320) & (v >= 0) & (v < 240)
    mask[v[ok].astype(int), u[ok].astype(int)] = True
    from scipy import ndimage
    mask = ndimage.binary_closing(ndimage.binary_dilation(mask, iterations=1), iterations=2)
    return canon, true, obs, ViewSilhouette(K, R_cw, t_cw, mask)


def test_alignment_recovers_pose_of_generated_object():
    canon, true, obs, view = _scene()
    # the "generator" returns the object at an arbitrary scale and heading, centred anywhere
    gen = canon.copy()
    gen.apply_scale(2.7)
    gen.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-110), (0, 1, 0)))
    gen.apply_translation((5, 3, -1))
    al = align_generated(gen.sample(8000), obs, [view], support_y=0.0)
    placed = gen.copy()
    placed.apply_transform(al.matrix)
    assert al.accepted, al.reason
    assert abs(al.scale * 2.7 - 0.8) < 0.08 * 0.8
    # recovered geometry overlaps the truth: mean surface distance < 3 cm
    from scipy.spatial import cKDTree

    d = cKDTree(true.sample(60000)).query(placed.sample(3000))[0]
    assert d.mean() < 0.03, d.mean()
    assert al.silhouette_iou > 0.7


def test_alignment_rejects_a_wrong_object():
    _, _, obs, view = _scene()
    wrong = trimesh.creation.cylinder(radius=0.05, height=1.0)  # thin pole: cannot explain an L
    al = align_generated(wrong.sample(4000), obs, [view], support_y=0.0)
    assert not al.accepted
