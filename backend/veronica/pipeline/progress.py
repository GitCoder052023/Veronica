from __future__ import annotations

import time
from typing import Callable

ProgressFn = Callable[[str, float, str], None]


def null_progress(stage: str, frac: float, msg: str = "") -> None:  # pragma: no cover - trivial
    pass


class Timer:
    """Collects wall-clock time per named stage for the manifest."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.timings[name] = round(timer.timings.get(name, 0.0) + time.perf_counter() - self.t, 3)

        return _Ctx()
