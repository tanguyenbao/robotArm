"""Low-poly 3D chess pieces (lathe profiles). White pieces are silver metal, black pieces are black metal."""
from __future__ import annotations

import math

from core.mesh import MATERIAL_BLACK, MATERIAL_SILVER, Mesh, mesh_box, mesh_lathe, mesh_merge, mesh_transform
from core.transforms import transform_rotation_x, transform_rotation_z, transform_translation

CHESS_PIECE_HEIGHT_MM = {"p": 38.0, "n": 52.0, "b": 58.0, "r": 46.0, "q": 66.0, "k": 76.0}
CHESS_PIECE_GRIP_HEIGHT_MM = {"p": 22.0, "n": 22.0, "b": 28.0, "r": 26.0, "q": 30.0, "k": 30.0}
CHESS_PIECE_GRIP_GAP_MM = {"p": 15.0, "n": 19.0, "b": 12.0, "r": 15.0, "q": 12.0, "k": 13.0}
_SEGMENTS = 10

_PROFILES = {
    "p": [(0, 0), (13, 0), (13, 4), (9, 6), (5, 13), (5.5, 18), (9, 20), (7, 22), (8.5, 27), (8.5, 31), (6, 35), (2.5, 37.5), (0, 38)],
    "r": [(0, 0), (14, 0), (14, 4), (10, 7), (7.5, 12), (7.5, 32), (11, 34), (11, 36), (12.5, 38), (12.5, 46), (9, 46), (9, 42), (0, 42)],
    "b": [(0, 0), (14, 0), (14, 4), (10, 7), (6, 16), (5, 28), (9, 30), (9, 32), (6, 34), (8, 42), (7, 50), (3, 56), (0, 58)],
    "q": [(0, 0), (15, 0), (15, 4), (11, 8), (6, 20), (5.5, 36), (10, 40), (10, 43), (7, 45), (12, 56), (12, 59), (6, 60), (4, 66), (0, 66)],
    "k": [(0, 0), (15, 0), (15, 4), (11, 8), (6, 20), (5.5, 38), (10, 42), (10, 45), (7, 47), (11, 55), (11, 58), (5, 60), (0, 62)],
}
_cache: dict[tuple[str, bool], Mesh] = {}


def _build(kind: str, is_white: bool) -> Mesh:
    material = MATERIAL_SILVER if is_white else MATERIAL_BLACK
    if kind == "n":
        base = mesh_lathe([(0, 0), (14, 0), (14, 4), (10, 7), (8, 12), (8, 22), (11, 24), (11, 26), (0, 26)], _SEGMENTS, material)
        head = mesh_box((13, 17, 26), (0, -1, 38), material)
        snout = mesh_transform(mesh_box((12, 15, 11), (0, 0, 0), material), transform_translation(0, 11, 36) @ transform_rotation_x(math.radians(-25)))
        ear = mesh_box((9, 5, 7), (0, -6, 53), material)
        mesh = mesh_merge([base, head, snout, ear])
        if not is_white:
            mesh = mesh_transform(mesh, transform_rotation_z(math.pi))
        return mesh
    parts = [mesh_lathe(_PROFILES[kind], _SEGMENTS, material)]
    if kind == "k":
        parts.append(mesh_box((4, 4, 15), (0, 0, 68), material))
        parts.append(mesh_box((12, 4, 4), (0, 0, 70), material))
    return mesh_merge(parts)


def chess_pieces_mesh(kind: str, is_white: bool) -> Mesh:
    """Mesh of one piece standing at the origin on z = 0. kind is one of p n b r q k."""
    key = (kind, is_white)
    if key not in _cache:
        _cache[key] = _build(kind, is_white)
    return _cache[key]
