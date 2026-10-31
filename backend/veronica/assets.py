"""Binary mesh / point packets consumed by the web client.

Layout ("VRNC" packet, little endian; readers also accept the legacy "STRK" magic):
    0   4  magic b"VRNC"
    4   4  uint32 header length N
    8   N  UTF-8 JSON header {"vertexCount", "indexCount", "attributes": {name: {offset, count, type, itemSize}}}
    ..     zero padding to a 4-byte boundary, then raw attribute buffers (each 4-byte aligned)

A custom container (instead of GLB) keeps the provenance attribute `observed` and
lets the browser build BufferGeometry with zero parsing beyond typed-array views.
GLB/OBJ/PLY export of the *edited* scene is done client-side.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .geometry.mesh import Mesh

MAGIC = b"VRNC"
LEGACY_MAGIC = b"STRK"  # scenes written before the VERONICA rename
_TYPES = {np.float32: "f32", np.uint32: "u32", np.uint8: "u8", np.uint16: "u16"}


def _pad4(b: bytes) -> bytes:
    return b + b"\0" * ((4 - len(b) % 4) % 4)


def write_packet(path: Path, arrays: dict[str, tuple[np.ndarray, int]], vertex_count: int, index_count: int = 0) -> int:
    header = {"vertexCount": int(vertex_count), "indexCount": int(index_count), "attributes": {}}
    blobs: list[bytes] = []
    offset = 0
    for name, (arr, item) in arrays.items():
        arr = np.ascontiguousarray(arr)
        t = _TYPES[arr.dtype.type]
        raw = _pad4(arr.tobytes())
        header["attributes"][name] = {"offset": offset, "count": int(arr.size), "type": t, "itemSize": item}
        blobs.append(raw)
        offset += len(raw)
    hj = _pad4(json.dumps(header).encode())
    data = MAGIC + struct.pack("<I", len(hj)) + hj + b"".join(blobs)
    path.write_bytes(data)
    return len(data)


def write_mesh(path: Path, mesh: Mesh) -> dict:
    arrays: dict[str, tuple[np.ndarray, int]] = {"position": (mesh.positions.astype(np.float32), 3)}
    if mesh.uvs is not None:
        arrays["uv"] = (mesh.uvs.astype(np.float32), 2)
    if mesh.colors is not None:
        arrays["color"] = (mesh.colors.astype(np.uint8), 3)
    if mesh.observed is not None:
        arrays["observed"] = (mesh.observed.astype(np.uint8), 1)
    arrays["index"] = (mesh.indices.astype(np.uint32).ravel(), 1)
    size = write_packet(path, arrays, mesh.n_vertices, mesh.indices.size)
    return {"file": path.name, "vertices": mesh.n_vertices, "faces": mesh.n_faces, "bytes": size, "texture": mesh.texture}


def write_points(path: Path, positions: np.ndarray, colors: np.ndarray) -> dict:
    size = write_packet(path, {"position": (positions.astype(np.float32), 3), "color": (colors.astype(np.uint8), 3)}, len(positions))
    return {"file": path.name, "points": int(len(positions)), "bytes": size}


def read_packet(path: Path) -> dict[str, np.ndarray]:
    """Inverse of write_packet (used by tests and the multi-view fuser)."""
    data = path.read_bytes()
    assert data[:4] in (MAGIC, LEGACY_MAGIC), "not a VRNC packet"
    (n,) = struct.unpack("<I", data[4:8])
    header = json.loads(data[8 : 8 + n].rstrip(b"\0"))
    base = 8 + n
    inv = {v: k for k, v in _TYPES.items()}
    out = {}
    for name, a in header["attributes"].items():
        dt = np.dtype(inv[a["type"]])
        arr = np.frombuffer(data, dtype=dt, count=a["count"], offset=base + a["offset"])
        out[name] = arr.reshape(-1, a["itemSize"]) if a["itemSize"] > 1 else arr
    return out
