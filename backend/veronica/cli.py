"""Command-line entry: `python -m veronica.cli reconstruct photo.jpg [--out DIR]`."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from .config import settings
from .pipeline.analysis import analyze, load_image
from .pipeline.build import build_scene


def main() -> None:
    ap = argparse.ArgumentParser(prog="veronica")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reconstruct", help="single photo → scene")
    r.add_argument("image", type=Path)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--depth-model", default=None)
    rb = sub.add_parser("rebuild", help="re-run the geometry stage from a scene's cached analysis")
    rb.add_argument("scene", type=Path)
    dl = sub.add_parser("download-models", help="pre-download model weights into the Hugging Face cache")
    dl.add_argument("--all", action="store_true", help="also the optional fallback depth model")
    dl.add_argument("--no-multiview", action="store_true", help="skip Depth Anything 3")
    sub.add_parser("doctor", help="report device, packages, model cache and optional components")
    args = ap.parse_args()

    if args.cmd == "download-models":
        return download_models(include_optional=args.all, multiview=not args.no_multiview)
    if args.cmd == "doctor":
        return doctor()

    if args.cmd == "rebuild":
        from .pipeline.analysis import Analysis

        an = Analysis.load(args.scene / "analysis.npz")
        old = json.loads((args.scene / "manifest.json").read_text())
        man = build_scene(an, args.scene, old["id"], lambda *a: None, old["source"].get("name", ""))
        for n in man["nodes"]:
            print(f'{n["id"]:22s} {n["provenance"]["observedFraction"]:.2f} {n["provenance"].get("completion")} {n["provenance"].get("completionParams", {}).get("freeSpaceCarved", "")}')
        return

    if args.cmd == "reconstruct":
        out = args.out or settings.scenes_dir / uuid.uuid4().hex[:10]
        sid = out.name
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        prog = lambda s, f, m="": print(f"[{f * 100:5.1f}%] {s:9s} {m}", flush=True)
        img, f = load_image(args.image)
        an = analyze(img, f, progress=prog, depth_model=args.depth_model)
        an.save(out / "analysis.npz")
        man = build_scene(an, out, sid, prog, args.image.name)
        print(json.dumps({"scene": str(out), "nodes": [(n["id"], n["name"], n["provenance"].get("completion")) for n in man["nodes"]],
                          "timings": man["pipeline"]["timings"], "total_s": round(time.perf_counter() - t0, 2)}, indent=1))



def _repos(include_optional: bool, multiview: bool) -> list[tuple[str, str]]:
    from .config import MODELS

    keys = [settings.depth_model, settings.detect_model, settings.segment_model]
    if include_optional:
        keys += [k for k in MODELS if k not in keys]
    out = [(MODELS[k].hf_id, MODELS[k].license) for k in dict.fromkeys(keys)]
    if multiview:
        from .providers.multiview import DA3_LICENSE, DA3_REPO

        out.append((DA3_REPO, DA3_LICENSE))
    return out


def download_models(include_optional: bool = False, multiview: bool = True) -> None:
    """Fetch weights only (no model is instantiated, so this needs no GPU and little RAM)."""
    from huggingface_hub import snapshot_download

    for repo, lic in _repos(include_optional, multiview):
        print(f"↓ {repo}  [{lic}]", flush=True)
        t = time.perf_counter()
        path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.yaml", "*.py"])
        print(f"  ✓ {time.perf_counter() - t:.0f}s  {path}", flush=True)
    print("All model weights are cached.")


def doctor() -> None:
    import platform
    import sys

    import torch

    from .pipeline.multiview import multiview_status
    from .providers.generative import CompleterClient

    ok = lambda b: "✓" if b else "✗"  # noqa: E731
    print(f"VERONICA doctor — Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    print(f"  torch {torch.__version__}   device in use: {settings.device}")
    print(f"  {ok(torch.backends.mps.is_available())} Apple GPU (MPS)   {ok(torch.cuda.is_available())} NVIDIA CUDA")
    mv = multiview_status()
    print(f"  {ok(mv['available'])} multi-view (Depth Anything 3){'' if mv['available'] else ' — ' + mv.get('reason', '')}")
    c = CompleterClient().health()
    state = "not configured (optional)" if not c.get("configured") else ("reachable" if c.get("ok") else f"unreachable: {c.get('reason') or c.get('error')}")
    print(f"  {ok(c.get('ok'))} generative completer: {state}")
    print(f"  data dir: {settings.data_dir}")
    try:
        from huggingface_hub import scan_cache_dir

        cached = {r.repo_id: r.size_on_disk for r in scan_cache_dir().repos}
    except Exception:
        cached = {}
    for repo, lic in _repos(False, mv["available"]):
        size = cached.get(repo)
        print(f"  {ok(size)} {repo}{f'  ({size / 2**30:.2f} GB)' if size else '  — run: python -m veronica.cli download-models'}")


if __name__ == "__main__":
    main()
