<p align="center"><img src="frontend/public/favicon.svg" width="72" alt="VERONICA"></p>

<h1 align="center">VERONICA</h1>
<p align="center"><b>The Spatial Intelligence System</b><br>
Open-source photo-to-3D reconstruction you can take apart and rearrange, by hand, in the browser.</p>

<p align="center">
<a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-5ee6d0"></a>
<img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-5ee6d0">
<img alt="Node 20+" src="https://img.shields.io/badge/node-20%2B-5ee6d0">
<img alt="Apple Silicon / CUDA / CPU" src="https://img.shields.io/badge/runs%20on-Apple%20Silicon%20%7C%20CUDA%20%7C%20CPU-5ee6d0">
</p>

VERONICA turns photographs into an editable, metric 3D scene of **separate objects**, rendered in the browser. You manipulate those objects with mouse and keyboard, or with your hands in the air through the webcam. Every surface it could not see is marked as inferred, never passed off as measured.

```
photo(s) ──► metric depth + camera ──► open-vocabulary detection ──► instance masks
                                                    │
         gravity-aligned room frame ◄── floor/wall planes (RANSAC)
                                                    │
   per object: observed surface (photo texture) + explicitly inferred hidden side
               (symmetry / revolution priors, carved against observed free space)
                                                    │
                     scene graph manifest + binary mesh packets
                                                    │
   browser: Three.js studio ── scene graph · gizmos · G/R/S · undo · export
            MediaPipe hands ── One-Euro filtering · pose state machine · gestures
```

Research and model selection, including licences and Apple-Silicon compatibility, are in [docs/RESEARCH.md](docs/RESEARCH.md).

## Quick start

### Native (recommended, and the only way to use the Apple GPU)

Requirements: macOS on Apple Silicon or Linux, [uv](https://docs.astral.sh/uv/), Node.js ≥ 20, and a webcam for hand control.

```bash
./scripts/install.sh     # Python 3.12 env (PyTorch with MPS/CUDA) + studio dependencies
```

```bash
./scripts/setup.sh       # creates .env, downloads model weights (~3.5 GB), runs a health check
```

```bash
./scripts/run.sh         # API on :8787 + studio on :5173, opens the browser; Ctrl-C stops both
```

Options:
- `install.sh --cpu` installs CPU-only PyTorch, for Linux without an NVIDIA GPU (much smaller).
- `install.sh --no-multiview` skips multi-photo support.
- `setup.sh --download-models` / `--skip-models` decides about the weights without asking.
- `run.sh --prod` serves an optimised build; `run.sh --no-open` doesn't open the browser.

Logs go to `.logs/`. To check your setup at any time:

```bash
cd backend && .venv/bin/python -m veronica.cli doctor
```

### Docker

```bash
./scripts/docker.sh up                 # studio on http://localhost:8080
```

```bash
./scripts/docker.sh up --gpu           # Linux + NVIDIA: CUDA PyTorch and GPU reservation
```

```bash
./scripts/docker.sh up --gpu --completer   # also run the TRELLIS completer worker
```

`./scripts/docker.sh down | logs | ps` manage the stack (it's plain `docker compose` underneath: `docker-compose.yml`, overlay `docker-compose.gpu.yml`).

- **Docker on macOS cannot use the Apple GPU**, so inference runs on the CPU and is several times slower. Use the native scripts on a Mac for speed.
- Give Docker ≥ 8 GB of memory.
- Model weights download on first use into the `veronica-models` volume. Set `HF_CACHE=/absolute/path/to/.cache/huggingface` in `.env` to reuse a host cache.
- Scenes persist in the `veronica-data` volume.

### Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `VERONICA_DEVICE` | auto | `mps`, `cuda` or `cpu` |
| `VERONICA_DEPTH_MODEL` | `depth-pro` | or `depth-anything-v2-metric-indoor-small` (Apache-2.0) |
| `VERONICA_MAX_SIDE` | `1280` | working image size; lower = faster and less memory |
| `VERONICA_DATA` | `backend/data` | where scenes are stored |
| `VERONICA_API_PORT` / `VERONICA_WEB_PORT` | `8787` / `5173` | native ports |
| `VERONICA_DOCKER_WEB_PORT` | `8080` | Docker studio port |
| `VERONICA_COMPLETER_URL` / `_TOKEN` | empty (off) | optional CUDA completer worker |
| `TORCH_BACKEND` | `cpu` | Docker API image: `cpu` or a CUDA build such as `cu126` |
| `HF_CACHE` | Docker volume | absolute path to share a host model cache with Docker |

Command-line reconstruction, without the studio:

```bash
cd backend && .venv/bin/python -m veronica.cli reconstruct ../samples/desk.png
```

## What you can do

**Reconstruction**
- **Single photo.** Depth Pro gives metric depth and focal length. Grounding DINO (~75 indoor classes, extendable) finds objects and SAM 2.1 segments them. The geometry stage then builds a gravity-aligned room frame, per-object meshes, and floor and walls.
- **Several photos of one place.** Depth Anything 3 recovers every camera pose and depth map in one pass, scaled to metres by Depth Pro. Objects are segmented in every view, matched across views in 3D, and **TSDF-fused**. Hidden sides are inferred only where *no* view saw them, and carved against every view's free space. Camera frustums and per-photo viewpoints appear in the viewer.
- **Refinement.** If the detector missed something, open *Source*, click the thing in the photo (SAM point prompt), or search for it by name. The scene rebuilds in about a second without re-running depth, and your edits are kept.

**Hidden sides without special hardware: more angles**
- On a Mac (or any machine without an NVIDIA GPU) the generative completer is **off**, and its controls are hidden. Real hidden sides come from **more photos**.
- Select several photos at once (3–8, walking around the scene in 20–40° steps with ~60 % overlap; the *How to shoot several angles* tips are in the Reconstruct panel). Or, on an existing scene, use **Add photos from other angles** in the inspector or scene panel. That rebuilds a new multi-view scene from the original photo(s) plus the new ones and keeps the old scene.
- iPhone HEIC photos are accepted. Portrait and landscape shots can be mixed; views are centre-cropped to a common frame, never stretched.

**Generative hidden sides (optional CUDA worker)**
- `completer-worker/` is a small service that wraps **TRELLIS** (MIT) on any Linux + NVIDIA (≥16 GB) machine (Docker recipe included). See [completer-worker/README.md](completer-worker/README.md).
- Disabled unless `VERONICA_COMPLETER_URL` (and optionally `VERONICA_COMPLETER_TOKEN`) is set. In the inspector, *Hidden side → Generate* (or *Generate hidden sides for all objects*) sends each object's masked crop to the worker; multi-view scenes send up to 4 views. The worker returns a textured mesh.
- VERONICA then **places** that mesh on the object's measured surface: a 360° heading search scored by surface distance and silhouette overlap in every view, a refit of the top hypotheses, height-anchored metric scale, and resting on its support. It then **carves** away everything any photo observed, so only the unseen side is kept and the observed surface stays the real photo geometry.
- Fits that don't match (silhouette IoU < 0.55, surface RMS too high, implausible footprint) are rejected, and the symmetry prior stays. Results persist across rebuilds and can be reverted per object.

**Scene editing (Blender-like)**
- Select by click (shift / ⌘ to add), ⌘-drag to box-select, or ⌥-click to pick inside a group.
- Gizmo move / rotate / scale, world or local axes, per-axis handle locks, snapping.
- Modal **G / R / S** with **X / Y / Z** constraints, Shift for precision, Esc to cancel.
- Numeric transform fields with drag-to-scrub, plus a uniform scale slider.
- Duplicate, delete, hide/show, lock, rename, **isolate**, **group / ungroup** (a real transform hierarchy), and reset position, rotation or scale.
- Scene-level rotate, scale and move, and **Restore original** to return to exactly what was reconstructed.
- Undo/redo for everything (snapshot history).
- Views: the photo viewpoint, iso, front, back, left, right, top, per-camera views, and frame-selection.
- Display modes: photo (unlit, because the texture already contains real light), clay, provenance (observed vs inferred), and normals. Representations: mesh + inferred, observed only, or point cloud.
- Export the **edited** scene as GLB (textured, hierarchy and transforms; inferred parts optional), OBJ (geometry + UVs, no texture), PLY (observed points with colour), a PNG screenshot, or WebM video (free or turntable).

### Hand control

Hand control is off until you enable it. Landmarks are computed **locally in the browser** (MediaPipe, GPU delegate), and no video leaves the machine.

| Gesture | Action | Why this gesture |
|---|---|---|
| Point (index out) | Aim the cursor; hover highlights | Low effort; distinct from every other pose |
| **Pinch tap** | Select / clear selection | The standard "click" on hand-tracked platforms |
| **Pinch-hold + move** | Drag the object in the view plane; push/pull the hand to move it in depth. On empty space: orbit | Direct-manipulation metaphor; depth comes from apparent hand size |
| **Fist + turn wrist** | Twist the selection in 3D (or the whole scene) | Uses the full hand orientation (world landmarks); a fist is never mistaken for a pinch |
| **Two-hand pinch** | Spread/close to scale, turn like a wheel to rotate, move to translate. With nothing selected: zoom/orbit the camera | Bimanual scaling is the most reliable way to change size |
| **Open palm (hold)** | Navigation: move to orbit, closer/further to zoom | Navigation is the "rest" pose, so it never edits anything |
| Open-palm **flick** | Next / previous view preset | Velocity-gated with a cooldown; slow motion orbits instead |
| **✌ held 0.65 s** | Toggle precision mode (0.3× gain, heavier smoothing) | A deliberate pose that is rarely held by accident |
| **Both palms open, 0.6 s** | Cancel the current operation, or undo | Hard to trigger accidentally; available from any state |

Modes (auto / navigate / object / scene) restrict what the hands can affect, and the HUD always shows the current operation and the next available gestures.

**Reliability engineering** (`frontend/src/gesture/`)
- One-Euro filtering of the cursor, palm and depth: heavy smoothing at rest, low lag in fast motion. Orientation uses frame-rate-independent slerp smoothing.
- Shape features use MediaPipe's **metric world landmarks** normalised by palm size, so they don't depend on hand size or distance. Position uses image landmarks.
- Per-finger extension hysteresis, pinch hysteresis (separate on/off thresholds), and time-based debouncing that works the same at 30 fps or 60 fps.
- A fist is never a pinch: a folded index finger vetoes the pinch.
- **Click-slip compensation.** Closing the fingers moves the pinch point, so a pinch anchors where the cursor was 110 ms *before* onset and then follows the **palm**. Release offsets ease out.
- Fist twist freezes as soon as the hand starts opening, so releasing doesn't twist the object back.
- Tracking dropouts under 260 ms are bridged. Longer losses cleanly end any grab, as a commit, never a stray move.
- Active-region mapping: the central ~75 % of the camera frame covers the whole screen, so you never reach for the edge of the camera's view.
- Per-user **pinch calibration** (open, then pinch), stored locally. A camera picker is included.

### Keyboard

| Keys | Action |
|---|---|
| `G` `R` `S`, then `X` `Y` `Z` | Grab / rotate / scale with axis constraint (`Enter` or click confirms, `Esc` cancels, `Shift` = fine) |
| `1` `2` `3` · `L` · `⇧Tab` · `⇧X/Y/Z` | Gizmo mode · local/world · snapping · axis handles |
| `⇧D` / `⌘D` · `X`/`Del` · `H` · `⌥H` · `I` | Duplicate · delete · hide · show all · isolate |
| `⌘G` / `⇧⌘G` · `⌥G` `⌥R` `⌥S` | Group / ungroup · clear location / rotation / scale |
| `F` · `C` · `M` · `P` · `⌘Z` / `⇧⌘Z` · `⌘A` · `Esc` | Frame · photo view · render mode · precision · undo / redo · select all · deselect |

## Architecture

```
backend/veronica/
  config.py               model registry (role, HF id, licence) + device selection
  providers/              depth (Depth Pro, DA-V2 metric), detect (Grounding DINO),
                          segment (SAM 2.1), multiview (Depth Anything 3), lazy model cache
  geometry/               camera, RANSAC planes + room-aligned world frame, grid meshing,
                          completion (mirror / revolution priors), mesh container
  pipeline/analysis.py    neural stage → Analysis (cached as analysis.npz)
  pipeline/build.py       deterministic geometry stage → manifest + VRNC packets
  pipeline/multiview.py   DA3 poses, cross-view association, TSDF fusion
  pipeline/refine.py      click/text add-object + fast rebuild
  pipeline/generate.py    generative completion: crops → worker → align → carve → persist
  geometry/align.py       place a generated canonical mesh on the measured surface
  providers/generative.py client for the CUDA completer worker
completer-worker/         TRELLIS service for a Linux/NVIDIA host (+ Dockerfile)
  api.py · jobs.py · cli.py · assets.py (binary packet format)
frontend/src/
  engine/                 VeronicaEngine (scene graph, picking, selection, manipulation
                          sessions, modal transforms, snapshot history, camera, capture),
                          provenance shader, exporters
  gesture/                HandTracker (MediaPipe), features, PoseClassifier, GestureEngine
                          (events, filtering), GestureController (gesture → engine)
  ui/                     panels, viewport HUD, onboarding, source view, shortcuts
```

**Interfaces that make upgrades cheap**
- Depth, detection, segmentation and multi-view are providers chosen by name. MoGe-2, VGGT, MapAnything or SAM 3 drop in behind the same signatures.
- Completion is a `Completer` protocol: `complete(ObjectContext) → Completion(mesh, method, params)`. A TRELLIS / SAM 3D / Hunyuan3D completer on a CUDA host only has to return a mesh aligned to the observed surface, and the carving and provenance machinery applies unchanged.
- Every gesture, gizmo and keyboard edit goes through one manipulation-session API: `beginManipulation → applyManipulation(worldDelta) → endManipulation(commit)`. That gives one undo model and one code path for multi-selection and hierarchy.
- Every scene manifest records the models, licences and stage timings that produced it.

## Validation (measured on Apple M5, 16 GB)

| Test | Result |
|---|---|
| Desk photo (512×384, your photo) | 7 objects + floor, wall, background in **≈17 s warm** (depth 2.4–4.7 s, detect ~5 s, segment 2–7 s, geometry < 1 s). Room-aligned frame. Desk 1.25 × 0.95 × 0.54 m, monitor 55 × 37 × 13 cm; "monitor rests on desk" found |
| Tabletop close-up (scikit-image *coffee*) | Table found as the support plane. Plate 4.9 cm tall. Mug (revolution completion, 10.9 cm tall) → rests on plate; spoon → rests on plate |
| Multi-view (3 views) | **≈39 s**: DA3 poses 2.8 s, metric scale, per-view segmentation. Desk, door, tower and drawer matched across all three views and TSDF-fused (fused desk 0.92 × 0.76 × 0.48 m vs 1.25 × 0.95 × 0.54 m from one view) |
| Rendering | ~60 fps at 1440×900 with ~380k triangles (BVH picking) |
| Hand tracking | GPU delegate, 3.3 ms per frame inference, 1.1 s model load |
| Gesture → engine (synthetic hands in the live app) | Pinch-grab drag moves the object ≈1:1 with the hand (0.164 vs 0.158 viewport). Two-hand 2× spread → 1.96× scale. 45° wrist roll → 44.9° object twist, kept on release. Every edit is one undo step |
| Generative completion (stand-in engine) | A cube returned at 3× scale, wrong heading and wrong place → aligned back to 0.5 m; only its back kept after carving; texture UVs correct; kept after a rebuild; revert restores the prior. A thin pole is rejected. L-shaped object: heading error 0–2° (worst ~18°), RMS 0.5–2 cm |
| Automated tests | `backend`: 17 pytest (synthetic ray-traced room: metric cube recovery, floor at y = 0, camera height, carving, revolve radius, API path safety, worker contract, alignment, generative end-to-end). `frontend`: 16 vitest (features, hysteresis, fist ≠ pinch, tap, click-slip, swipe cooldown, hand loss, filters, history, packets) |

Run them with:

```bash
cd backend && uv run pytest
```

```bash
cd frontend && npx vitest run
```

**Not yet validated:** live webcam gestures with real hands (the gesture logic is exercised with synthetic landmark streams; please try it), and real multi-photo captures of bedrooms, living rooms and kitchens. The multi-view runs above used overlapping crops of one photo, which tests the machinery but not true parallax. In one such run (*desk + 2 angles*) the door, which is cut off differently in each crop, ended up as three separate objects. Cross-view matching thresholds need tuning on real multi-angle photos.

## Honest limitations

- **Single-photo hidden sides are guesses.** They come from symmetry and revolution priors, or from TRELLIS when a CUDA worker is attached, and are always marked. A generator gives far better shapes for asymmetric objects, but it is still a guess. Heading can be ambiguous from one view (single-view alignment tests: 0–2° error on most headings, up to ~18° on the hardest), and more photos fix that.
- **The CUDA worker is untested on real hardware.** No NVIDIA GPU was available while building it. Its HTTP contract, VERONICA's alignment, carving, persistence and UI were verified with a stand-in engine; the TRELLIS calls follow TRELLIS's own example and pipeline source.
- **Absolute scale** comes from monocular depth (±10–20 %). Relative layout is better.
- **Thin and see-through parts.** Cables, chair legs and glass are hard for monocular depth. See-through gaps inside a mask are returned to the room planes when they lie on them.
- **Mask quality bounds object quality.** Missed objects stay part of the background until you add them with a click or by name.
- **Depth Pro weights are research-only** (Apple ML Research licence). For commercial use, pick the Apache-licensed Depth Anything V2 metric model in *Options*. DA3-BASE, Grounding DINO, SAM 2.1, MediaPipe and three.js are permissively licensed.
- **No Gaussian splats.** Training them needs CUDA (or Brush), and splats are not object-editable.

## Roadmap

1. More completer engines behind the same worker API: TRELLIS.2 (24 GB) and SAM 3D Objects (gated, 32 GB), which also predicts layout.
2. SAM 3 text-prompted segmentation once access to the gated weights is granted.
3. Exporting the multi-view poses as COLMAP → Brush splat training → an optional splat layer (Spark) under the editable objects.
4. LaMa inpainting for background textures.

## Contributing

Contributions are welcome: providers (depth, segmentation, multi-view, completion), gesture ergonomics, rendering, and real-world test captures. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

VERONICA's source code is licensed under the [Apache License 2.0](LICENSE).

The **models it downloads at runtime are not part of this repository** and keep their own licences, listed in [NOTICE](NOTICE). In particular, the default single-view depth model (Apple Depth Pro) is research-only; select the Apache-licensed Depth Anything V2 model for commercial use.

