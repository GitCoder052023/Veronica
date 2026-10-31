# VERONICA completer worker (CUDA)

Generates the **unseen side** of reconstructed objects with
[TRELLIS](https://github.com/microsoft/TRELLIS) (MIT code and weights). It needs a Linux
machine with an NVIDIA GPU with ≥16 GB VRAM (TRELLIS is verified by its authors on
A100 and A6000). The VERONICA backend on your Mac sends each object's masked crop(s)
and receives a GLB, then aligns it to the object's measured surface and keeps only the
parts no photo observed.

```bash
docker build -t veronica-completer .
docker run --gpus all -p 8790:8790 -e VERONICA_COMPLETER_TOKEN=change-me \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface veronica-completer
# on the Mac:
VERONICA_COMPLETER_URL=http://gpu-box:8790 VERONICA_COMPLETER_TOKEN=change-me \
  uv run --extra multiview uvicorn veronica.api:app --port 8787
```

API (`veronica-completer/1`):

| Endpoint | |
|---|---|
| `GET /v1/health` | `{ok, model, device, vram_gb}` |
| `POST /v1/complete` | multipart `images` (1–4 RGBA PNGs; alpha = object mask; several = several views of one object) + `params` JSON (`seed`, `ss_steps`, `ss_cfg`, `slat_steps`, `slat_cfg`, `simplify`, `texture_size`, `multi_algo`) → `model/gltf-binary`, `X-Veronica-Meta` header with timing |

Without Docker: install TRELLIS with its `setup.sh` (see its README), then
`pip install -r requirements.txt && uvicorn worker:app --host 0.0.0.0 --port 8790`
from a shell where `trellis` is importable.

Licences: TRELLIS code and weights are MIT. Its GLB texture baking renders Gaussians
with `diff-gaussian-rasterization` (the `--mipgaussian` submodule), which carries the
Inria **non-commercial** licence; check this before commercial use.

Privacy: only object crops are sent, and only to the URL you configure. Set a token
when the worker is reachable from a network.
