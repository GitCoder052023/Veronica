# VERONICA: technology research (Stage 1)

Verified 2026-11-01 against the repositories' LICENSE files, Hugging Face model cards, and the PyPI/npm registries. Rows marked **ran here** were also installed and executed on the development machine (Apple M5, 16 GB unified memory, macOS, PyTorch MPS, no CUDA).

## 1. What each technology actually does

| Project | What it really outputs | Fits VERONICA as | Licence (code / weights) | Runs on Apple Silicon? | Verdict |
|---|---|---|---|---|---|
| **DUSt3R** (naver) | Pairwise point maps, then globally aligned point clouds and camera poses from unposed photos | Multi-view geometry and poses | CC BY-NC-SA 4.0 / CC BY-NC-SA 4.0 (+ dataset licences) | CPU only (optional CUDA RoPE kernel); slow | Superseded. Non-commercial. Not used |
| **MASt3R** (naver) | DUSt3R plus dense matching features and metric scale | Same | CC BY-NC-SA 4.0 | Same | Not used, for the same reasons |
| **3D Gaussian Splatting** (graphdeco) | Per-scene optimised Gaussians (photoreal novel views). Needs posed multi-view input and minutes of CUDA training | Photoreal visualisation | Inria/MPII **non-commercial** | No (CUDA rasteriser) | Not used. Splats are not editable objects |
| **gsplat** (nerfstudio; the `g-splat` URL in the brief 404s) | CUDA splat rasteriser and trainer | Same | Apache-2.0 | **No** (CUDA JIT) | Not usable on this Mac |
| **Brush** | 3DGS training in Rust/wgpu, native on Metal and in WebGPU browsers | Future splat training on the Mac | Apache-2.0 | Yes | Roadmap: export COLMAP from the multi-view poses and train in Brush |
| **LGM** (3DTopia) | Image → multi-view diffusion → Gaussians (object-level) | Object completion | MIT / MIT | No (xformers, diff-gaussian-rasterization, nvdiffrast) | Stale (2024), CUDA only |
| **InstantMesh** (TencentARC) | Image → multi-view → LRM mesh (object-level) | Object completion | Apache-2.0 | No (xformers, nvdiffrast) | Stale; superseded by TRELLIS |
| **TRELLIS / TRELLIS.2** (Microsoft) | Image → textured mesh / Gaussians of a **single isolated object** in canonical pose | Object completion | MIT / MIT (TRELLIS.2 needs gated DINOv3) | No: spconv, flash-attn, kaolin; ≥16–24 GB NVIDIA | **Used** as the generative completer, via `completer-worker/` on a CUDA host |
| **SAM 3D Objects** (Meta, 2025-11) | Per-object shape, texture **and pose in the scene** from one image plus masks | The closest match to "object-level single-image reconstruction" | SAM License, **gated** | No: Linux and NVIDIA ≥32 GB | Ideal remote completer when a big GPU exists |
| **Hunyuan3D-2 / 2.1** | Untextured mesh (DiT), then PBR texture | Object completion | Tencent community licence: **excluded in the EU, UK and South Korea** | Shape gen has MPS paths; texturing needs compiled rasterisers | — |
| **Hunyuan3D-2mini** (turbo) — **ran here** | Untextured mesh from one image (0.6B DiT, 5 steps) | Object completion on the Mac | Same Tencent community licence (territory exclusion surfaced in install and UI) | **Yes: 17–33 s per object on MPS fp16, 8.7 GB peak**, after casting float64 scheduler tensors and replacing the decoder (hy3dgen's hierarchical/FlashVDM decoders find no surface on MPS) | **Used** as the local completer on Apple Silicon; colour borrowed from the photo |
| **TRELLIS.2 Mac port** (`trellis-mac`) | TRELLIS.2 on MPS, textured | Object completion | MIT port; DINOv3 gated, RMBG-2.0 CC BY-NC | Peaks ~18 GB, 24 GB+ recommended; ~5 min on an M4 Pro | Not used: does not fit 16 GB |
| **Marigold** | Affine-invariant depth / normals by diffusion (SD2 UNet, ~866M) | Depth | Apache-2.0 / OpenRAIL++ | Yes, but seconds per denoising step | Not metric. Slower than feed-forward depth with no gain here |
| **Depth Pro** (Apple) — **ran here** | **Metric** depth **and focal length** from one image | Single-view metric geometry | Code permissive; **weights apple-amlr (research only)** | **Yes: 2.4 s at 512 px on MPS fp16** | **Default single-view depth** (licence surfaced in the UI and manifest) |
| **Depth Anything V2** (metric indoor small) | Metric depth, no focal | Commercially clean fallback | Small: Apache-2.0 (Base/Large: CC BY-NC) | Yes | **Fallback provider** (selectable) |
| **Depth Anything 3** — **ran here** | **Any-view**: per-view depth, confidence, intrinsics **and extrinsics** from 1..N unposed images | Multi-view poses and depth (replaces DUSt3R/MASt3R) | Apache-2.0 for **DA3-SMALL/BASE**/METRIC-L/MONO-L; CC BY-NC for L/G | **Yes: 0.23 s for 2 views at 504 px** (pip pulls CUDA-only xformers, but the code falls back) | **Multi-view engine** |
| **MoGe-2** (Microsoft) | Metric point map, intrinsics, normals | Strong single-view alternative | MIT / MIT | v2 yes; **main branch (v3) adds a CUDA-only dep** | Not used (our git-install route was blocked); good future provider pinned at `b942f00b` |
| **VGGT / MapAnything** (Meta) | Feed-forward multi-view poses, depth, point maps | Multi-view | VGGT-1B CC BY-NC (commercial ckpt gated); MapAnything has an Apache checkpoint | 1.2B params: tight on 16 GB | Alternatives to DA3 behind the same interface |
| **Grounding DINO** — **ran here** | Open-vocabulary boxes from text prompts | Object discovery | Apache-2.0 | Yes (~1 s per prompt chunk) | **Detector** |
| **SAM 2.1** — **ran here** | Box/point-prompted masks | Instance masks | Apache-2.0 | Yes | **Segmenter**, and click-to-add-object |
| **SAM 3** | Text-prompted instance segmentation (concepts) | Would merge detection and segmentation | SAM License, **gated** | In transformers 5.22; MPS untested | Upgrade path once the user has gated access |
| **OWLv2 / Florence-2** | Open-vocabulary detection | Alternative detectors | Apache-2.0 / MIT | Yes | Interchangeable; not needed |
| **mkkellogg/GaussianSplats3D** | Three.js splat viewer | Browser splats | MIT | Browser | **Deprecated by its author in favour of Spark** |
| **Spark** (`@sparkjsdev/spark` 2.4) | Three.js splat renderer (WebGL2) | Browser splats | MIT | Browser | Roadmap, for importing external splats |
| **MediaPipe HandLandmarker** (`@mediapipe/tasks-vision` 1.2.0) — **ran here** | 21 image landmarks + 21 metric world landmarks per hand, handedness, 2 hands | Gesture input | Apache-2.0 | **Browser GPU delegate: 3.3 ms per frame, 1.1 s load** | **Hand tracker** |
| **three.js r187** | WebGL2/WebGPU renderer, TransformControls (r169+ needs `getHelper()`) | Rendering, gizmos, export | MIT | Browser | **Renderer** |
| **three-mesh-bvh** | BVH-accelerated raycasting | Picking ~1M triangles every frame | MIT | Browser | **Used** |

## 2. Conclusions that shaped the architecture

1. **A Gaussian splat is not an editable object.** Splats need posed multi-view capture and CUDA training, and they produce one undifferentiated cloud. They are excellent for photoreal viewing and useless for "select the bottle and move it". VERONICA's editable unit is the **per-object mesh**. Splats can come later as an optional *visual* layer.
2. **Object generators (TRELLIS, LGM, InstantMesh, Hunyuan3D) reconstruct one isolated object in a canonical frame.** They don't preserve where the object sits in the room. Used alone they violate principle 3 ("individually generated objects that don't preserve spatial relationships"). The right role for them is **completion of an object already placed by metric geometry**, aligned to its observed surface. Most need CUDA. VERONICA therefore defines a completer interface and ships two explicit, inspectable geometric completers, plus a generative worker: TRELLIS on CUDA, and Hunyuan3D-2mini, which runs on the Mac's GPU.
3. **Single-image geometry has to be metric and needs a focal length.** Depth Pro provides both, so objects get real-world sizes, and the scene can be viewed from exactly the photo's viewpoint and then orbited.
4. **DUSt3R/MASt3R are superseded for this use.** Depth Anything 3 does the same job (poses plus depth from unposed photos) in a single forward pass, under Apache-2.0 for the base model, and it runs on MPS.
5. **Browser vs backend.** Hand tracking (3 ms per frame) and all rendering and interaction run in the browser. Depth, detection, segmentation and fusion run on the Python backend, here on the Mac's GPU via MPS. The browser could run some of these through ONNX/WebGPU. But Depth Pro (950M params) and DA3 at useful resolutions are seconds-scale even natively, and doing them in-browser would trade quality for a slogan (principle 6).

## 3. What a single photo can and cannot give you

Observed is the camera-facing surface of each object, at the photo's resolution and with metric depth error of roughly 10–20 % in absolute scale. Relative shape is much better than absolute scale.

Not observed are backs, undersides, and whatever sits behind objects. VERONICA fills these with *named priors* and labels every invented vertex `observed = 0`:

- **Mirror symmetry** across the object's gravity-aligned vertical symmetry planes. The depth extent comes from visible side faces when present, otherwise from a per-class depth/width prior, and the manifest says which.
- **Surface of revolution** for rotationally symmetric classes (bottles, mugs, plates, vases). The axis comes from a circle fit to the visible arc.
- **Free-space carving.** Any inferred surface that the camera would have seen is deleted, because the photo proves that space is empty. In multi-view mode this check runs against every view.
- **Room planes continued behind objects** by ray–plane intersection (rooms are convex from inside). Other background holes are filled by inpainting.

The UI hatches every inferred surface, can hide inferred geometry entirely, and reports the observed/inferred area split per object.
