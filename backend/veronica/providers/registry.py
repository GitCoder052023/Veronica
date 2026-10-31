"""Lazy, thread-safe model cache. Models load on first use and stay resident until
`unload_all()`; the pipeline holds a single inference lock so two jobs never fight
over unified memory on a 16 GB machine."""
from __future__ import annotations

import gc
import threading
from typing import Any, Callable

import torch

_cache: dict[str, Any] = {}
_load_lock = threading.Lock()
inference_lock = threading.Lock()


def get_or_load(key: str, loader: Callable[[], Any]) -> Any:
    with _load_lock:
        if key not in _cache:
            _cache[key] = loader()
        return _cache[key]


def loaded() -> list[str]:
    return sorted(_cache)


def unload_all() -> None:
    with _load_lock:
        _cache.clear()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
