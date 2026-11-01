# VERONICA completer worker

Generates the **unseen side** of reconstructed objects. The VERONICA backend sends each
object's masked crop(s) and receives a mesh, then aligns it to the object's measured
surface and keeps only the parts no photo observed. Two engines speak the same API;
`VERONICA_COMPLETER_ENGINE` picks one (`auto`, the default, uses whichever is installed).

## On a Mac: Hunyuan3D-2mini (`hunyuan3d-mini`)

Pure PyTorch, so it runs on the Apple GPU through Metal (MPS). `scripts/install.sh`
installs it into `completer-worker/.venv` on Apple Silicon and `scripts/run.sh` starts it
on `127.0.0.1:8790`, so nothing needs configuring. By hand:

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements-mac.txt \
  && uv pip install --python .venv/bin/python --no-deps hy3dgen==2.0.2
VERONICA_COMPLETER_ENGINE=hunyuan3d-mini .venv/bin/uvicorn worker:app --host 127.0.0.1 --port 8790
```

- Model: `tencent/Hunyuan3D-2mini`, `hunyuan3d-dit-v2-mini-turbo` (5 steps; set
  `HUNYUAN_SUBFOLDER` for another variant). Only the safetensors weights are fetched
  (~3.8 GB).
- Measured on an M5 with 16 GB: ~27 s model load, 17–33 s per object, ~8.7 GB peak.
- Untextured output: the backend colours it from the photo.
- Single-image conditioning: with several views, the first (largest) is used.
- Two Metal-specific changes, both in `worker.py`: the scheduler's float64 tensors are cast to
  float32, and the occupancy volume is decoded coarse-to-fine by the worker itself. hy3dgen's
  hierarchical and FlashVDM decoders find no surface on MPS, and its dense decoder takes ~70 s.
- `VERONICA_COMPLETER_IDLE_UNLOAD=<seconds>` frees the model when idle, which matters on
  unified memory.
- **Licence:** the weights are under the *Tencent Hunyuan 3D 2.0 Community License*, which
  does not grant rights in the EU, the UK or South Korea, and has further use conditions.
  Read it (`LICENSE` in the model repository) before use.

## On Linux + NVIDIA: TRELLIS (`trellis`)

[TRELLIS](https://github.com/microsoft/TRELLIS) (MIT code and weights), textured and
multi-view conditioned. It needs an NVIDIA GPU with ≥16 GB VRAM (TRELLIS is verified by
its authors on A100 and A6000).

```bash
docker build -t veronica-completer .
docker run --gpus all -p 8790:8790 -e VERONICA_COMPLETER_TOKEN=change-me \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface veronica-completer
# on the machine running VERONICA (this overrides a local completer):
VERONICA_COMPLETER_URL=http://gpu-box:8790 VERONICA_COMPLETER_TOKEN=change-me ./scripts/run.sh
```

API (`veronica-completer/1`):

| Endpoint | |
|---|---|
| `GET /v1/health` | `{ok, engine, model, license, device, vram_gb, loaded}`; does not load the model |
| `POST /v1/complete` | multipart `images` (1–4 RGBA PNGs; alpha = object mask; several = several views of one object) + `params` JSON (`seed`; TRELLIS: `ss_steps`, `ss_cfg`, `slat_steps`, `slat_cfg`, `simplify`, `texture_size`, `multi_algo`; Hunyuan: `steps`, `guidance`, `octree_resolution`, `max_faces`) → `model/gltf-binary`, `X-Veronica-Meta` header with timing |

Without Docker: install TRELLIS with its `setup.sh` (see its README), then
`pip install -r requirements.txt && uvicorn worker:app --host 0.0.0.0 --port 8790`
from a shell where `trellis` is importable.

Licences: TRELLIS code and weights are MIT (Hunyuan3D-2mini: see above). Its GLB texture baking renders Gaussians
with `diff-gaussian-rasterization` (the `--mipgaussian` submodule), which carries the
Inria **non-commercial** licence; check this before commercial use.

Privacy: only object crops are sent, and only to the URL you configure. Set a token
when the worker is reachable from a network.
