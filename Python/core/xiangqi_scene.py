"""Meshes of the xiangqi (Chinese chess) board and its pieces, and the scene builder for the viewport.

Pieces are flat discs (red or black metal) with the piece name raised on top. The names come in as small on/off bitmaps
(`glyphs`), drawn by the adapter layer from a system font, so this module stays free of Qt. Names read upright when the
viewer stands on the arm side of the table.
"""
from __future__ import annotations

import math
import re

import numpy as np

from core.arm_model import ArmGeometry
from core.arm_visual import arm_visual_meshes
from core.mesh import (MATERIAL_BLACK, MATERIAL_BOARD_DARK, MATERIAL_BOARD_LIGHT, MATERIAL_SILVER, Mesh, mesh_box,
                       mesh_lathe, mesh_merge, mesh_plate, mesh_transform)
from core.transforms import transform_rotation_z, transform_translation
from core.world_scene import WorldScene, world_scene_floor_meshes, world_scene_tool_position
from core.xiangqi_board import (XIANGQI_BOARD_CENTER_MM, XIANGQI_CELL_MM, XIANGQI_DISC_HEIGHT_MM, XIANGQI_FILES,
                                XIANGQI_PIECE_GRIP_HEIGHT_MM, XIANGQI_PIECE_RADIUS_MM, XIANGQI_RANKS,
                                xiangqi_board_point_xy)
from core.xiangqi_world import XiangqiWorld

_MARGIN_MM = 18.0
_LINE_MM = 1.2
_RIM_MM = 14.0
_DISC_SEGMENTS = 20
_GLYPH_RELIEF_MM = 0.8
_GLYPH_SIZE_PER_RADIUS = 1.45
_MARKER_RADIUS_MM = 17.5
_MARKER_HEIGHT_MM = 0.8
_LINE_COLOR = (0.16, 0.18, 0.22)
_RIVER_COLOR = (0.66, 0.74, 0.86)
_RED_COLOR = (0.74, 0.14, 0.12)
_GLYPH_COLOR = (0.93, 0.94, 0.96)

_board_cache: list[list[list[Mesh]]] = []
_piece_cache: dict[tuple[str, bool], Mesh] = {}
_marker_cache: dict[tuple, Mesh] = {}


def _line(start: tuple[float, float], end: tuple[float, float]) -> Mesh:
    """Thin flat strip on the board surface from start to end."""
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    strip = mesh_plate(length + _LINE_MM, _LINE_MM, (0.0, 0.0, 0.0), MATERIAL_BOARD_DARK, _LINE_COLOR)
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, 0.0)
    return mesh_transform(strip, transform_translation(*middle) @ transform_rotation_z(angle))


def _board_lines() -> Mesh:
    last_file, last_rank = XIANGQI_FILES - 1, XIANGQI_RANKS - 1
    lines = [_line(xiangqi_board_point_xy(0, rank), xiangqi_board_point_xy(last_file, rank)) for rank in range(XIANGQI_RANKS)]
    for file_index in range(XIANGQI_FILES):
        if file_index in (0, last_file):  # the edge files run through the river
            lines.append(_line(xiangqi_board_point_xy(file_index, 0), xiangqi_board_point_xy(file_index, last_rank)))
        else:
            lines.append(_line(xiangqi_board_point_xy(file_index, 0), xiangqi_board_point_xy(file_index, 4)))
            lines.append(_line(xiangqi_board_point_xy(file_index, 5), xiangqi_board_point_xy(file_index, last_rank)))
    for first_rank in (0, 7):  # the two palaces, each with its crossing diagonals
        lines.append(_line(xiangqi_board_point_xy(3, first_rank), xiangqi_board_point_xy(5, first_rank + 2)))
        lines.append(_line(xiangqi_board_point_xy(5, first_rank), xiangqi_board_point_xy(3, first_rank + 2)))
    return mesh_merge(lines)


def _board_background() -> list[list[Mesh]]:
    """Layers drawn in order: floor, slab, surface, river, lines (coplanar parts get their own layer).

    The board looks the same turned by 180 degrees, so it needs no flipped variant."""
    if not _board_cache:
        center_x, center_y = XIANGQI_BOARD_CENTER_MM
        width = (XIANGQI_FILES - 1) * XIANGQI_CELL_MM
        height = (XIANGQI_RANKS - 1) * XIANGQI_CELL_MM
        slab = mesh_box((width + 2 * _MARGIN_MM + 2 * _RIM_MM, height + 2 * _MARGIN_MM + 2 * _RIM_MM, 15.6),
                        (center_x, center_y, -8.2), MATERIAL_BLACK)
        surface = mesh_plate(width + 2 * _MARGIN_MM, height + 2 * _MARGIN_MM, (center_x, center_y, 0.0), MATERIAL_BOARD_LIGHT)
        river = mesh_plate(width, XIANGQI_CELL_MM - _LINE_MM, (center_x, center_y, 0.0), MATERIAL_BOARD_LIGHT, _RIVER_COLOR)
        _board_cache.append([world_scene_floor_meshes(), [slab], [surface], [river], [_board_lines()]])
    return _board_cache[0]


def _board_rim() -> list[Mesh]:
    center_x, center_y = XIANGQI_BOARD_CENTER_MM
    inner_x = (XIANGQI_FILES - 1) * XIANGQI_CELL_MM / 2 + _MARGIN_MM
    inner_y = (XIANGQI_RANKS - 1) * XIANGQI_CELL_MM / 2 + _MARGIN_MM
    return [
        mesh_box((2 * inner_x + 2 * _RIM_MM, _RIM_MM, 3), (center_x, center_y - inner_y - _RIM_MM / 2, 0.6), MATERIAL_SILVER),
        mesh_box((2 * inner_x + 2 * _RIM_MM, _RIM_MM, 3), (center_x, center_y + inner_y + _RIM_MM / 2, 0.6), MATERIAL_SILVER),
        mesh_box((_RIM_MM, 2 * inner_y, 3), (center_x - inner_x - _RIM_MM / 2, center_y, 0.6), MATERIAL_SILVER),
        mesh_box((_RIM_MM, 2 * inner_y, 3), (center_x + inner_x + _RIM_MM / 2, center_y, 0.6), MATERIAL_SILVER),
    ]


def _glyph_rectangles(rows: tuple[str, ...]) -> list[tuple[int, int, int, int]]:
    """Merge the '#' cells of a bitmap into few rectangles: (first column, end column, first row, end row)."""
    rectangles: list[tuple[int, int, int, int]] = []
    open_runs: dict[tuple[int, int], int] = {}
    for row_index in range(len(rows) + 1):
        runs = {(match.start(), match.end()) for match in re.finditer(r"#+", rows[row_index])} if row_index < len(rows) else set()
        for run in [run for run in open_runs if run not in runs]:
            rectangles.append((run[0], run[1], open_runs.pop(run), row_index))
        for run in runs:
            open_runs.setdefault(run, row_index)
    return rectangles


def xiangqi_scene_piece_mesh(kind: str, is_red: bool, glyph: tuple[str, ...]) -> Mesh:
    """One piece standing at the origin on z = 0. kind is one of K A E H R C P, glyph is the bitmap of its name."""
    key = (kind, is_red)
    if key not in _piece_cache:
        radius = XIANGQI_PIECE_RADIUS_MM[kind]
        height = XIANGQI_DISC_HEIGHT_MM
        material = MATERIAL_SILVER if is_red else MATERIAL_BLACK
        parts = [mesh_lathe([(0, 0), (radius, 0), (radius, height - 2), (radius - 2, height), (0, height)],
                            _DISC_SEGMENTS, material, _RED_COLOR if is_red else None)]
        cells = len(glyph)
        pitch = radius * _GLYPH_SIZE_PER_RADIUS / cells if cells else 0.0
        for first_column, end_column, first_row, end_row in _glyph_rectangles(glyph):
            parts.append(mesh_box(((end_column - first_column) * pitch, (end_row - first_row) * pitch, _GLYPH_RELIEF_MM),
                                  (((first_column + end_column) / 2 - cells / 2) * pitch,
                                   (cells / 2 - (first_row + end_row) / 2) * pitch,
                                   height + _GLYPH_RELIEF_MM / 2), MATERIAL_SILVER, _GLYPH_COLOR))
        _piece_cache[key] = mesh_merge(parts)
    return _piece_cache[key]


def _marker_mesh(color: tuple) -> Mesh:
    if color not in _marker_cache:
        _marker_cache[color] = mesh_lathe([(0, 0), (_MARKER_RADIUS_MM, 0), (_MARKER_RADIUS_MM, _MARKER_HEIGHT_MM),
                                           (0, _MARKER_HEIGHT_MM)], 16, MATERIAL_SILVER, color)
    return _marker_cache[color]


def xiangqi_scene_build(world: XiangqiWorld, geometry: ArmGeometry, joints_rad: np.ndarray, gripper_gap_mm: float,
                        highlights: dict[int, tuple], glyphs: dict[tuple[str, bool], tuple[str, ...]]) -> WorldScene:
    """Meshes for one frame of the xiangqi table. highlights maps a square index to an rgb colour."""
    tool = world_scene_tool_position(geometry, joints_rad)
    static_objects: list[Mesh] = _board_rim()
    dynamic_objects: list[Mesh] = arm_visual_meshes(geometry, joints_rad, gripper_gap_mm)
    for square, color in sorted(highlights.items()):
        x_mm, y_mm = world.square_xy(square)
        static_objects.append(mesh_transform(_marker_mesh(color), transform_translation(x_mm, y_mm, 0.0)))
    standing = []
    for piece in world.pieces.values():
        mesh = xiangqi_scene_piece_mesh(piece.kind, piece.is_white, glyphs.get((piece.kind, piece.is_white), ()))
        if piece.held:
            position = (tool[0], tool[1], tool[2] - XIANGQI_PIECE_GRIP_HEIGHT_MM[piece.kind] + 7.0)
            dynamic_objects.append(mesh_transform(mesh, transform_translation(*position)))
        else:
            static_objects.append(mesh_transform(mesh, transform_translation(piece.x_mm, piece.y_mm, 0.0)))
            standing.append((piece.piece_id, piece.kind, round(piece.x_mm, 1), round(piece.y_mm, 1)))
    highlight_key = tuple(sorted(highlights.items()))
    return WorldScene(_board_background(), static_objects, dynamic_objects,
                      ("xiangqi", highlight_key, tuple(sorted(standing))))
