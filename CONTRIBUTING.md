# Contributing to VERONICA

Thanks for helping build VERONICA — The Spatial Intelligence System.

## Development setup

```bash
./scripts/install.sh      # backend (uv) + frontend (npm) dependencies
./scripts/setup.sh        # .env + optional model pre-download
./scripts/run.sh          # API on :8787, studio on :5173
```

Docker alternative: `docker compose up --build` (see README → Docker).

## Tests

```bash
cd backend && uv run pytest              # geometry, pipeline, API, alignment, completer contract
cd frontend && npx tsc -b && npx vitest run && npm run lint
```

Backend tests use a ray-traced synthetic room and need no model weights. Please add a test with every behaviour change; the synthetic room (`backend/tests/conftest.py`) and the synthetic hand (`frontend/src/gesture/__tests__/synthHand.ts`) make most changes testable without hardware.

## Where things live

| Area | Path |
|---|---|
| Model providers (depth, detection, segmentation, multi-view, generative client) | `backend/veronica/providers/` |
| Geometry (planes, meshing, completion priors, alignment) | `backend/veronica/geometry/` |
| Pipelines (single-view, multi-view, refine, generative) | `backend/veronica/pipeline/` |
| 3D engine (scene graph, manipulation, history, export) | `frontend/src/engine/` |
| Gesture recognition and mapping | `frontend/src/gesture/` |
| CUDA completer worker | `completer-worker/` |

## Principles for changes

1. **Never present inferred geometry as observed.** Anything not seen in a photo carries `observed = 0` and is shown as inferred.
2. **Metric and gravity-aligned.** Keep the world frame Y-up in metres.
3. **Every edit goes through the manipulation session API**, so undo/redo and multi-selection keep working.
4. **Licences.** New models go in `veronica/config.py` with their licence, and in `NOTICE`.

## Pull requests

- Keep PRs focused; describe what you measured (timings, accuracy) when touching the pipeline.
- Run the test commands above.
- By contributing you agree that your contributions are licensed under the Apache License 2.0.

Please be respectful; we follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
