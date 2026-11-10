"""Builds every mesh of the scene (floor, board, pieces, arm) for one frame."""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from core.arm_model import ArmGeometry, arm_model_frames
from core.arm_visual import arm_visual_meshes
from core.chess_pieces import CHESS_PIECE_GRIP_HEIGHT_MM, chess_pieces_mesh
from core.chess_world import CHESS_BOARD_CENTER_MM, CHESS_SQUARE_MM, ChessWorld
from core.mesh import (MATERIAL_BLACK, MATERIAL_BOARD_DARK, MATERIAL_BOARD_LIGHT, MATERIAL_GROUND, MATERIAL_SILVER,
                       Mesh, mesh_box, mesh_merge, mesh_plate, mesh_transform)
from core.transforms import transform_translation

_FLOOR_MM = -16.0
_floor_cache: list[Mesh] = []
_board_cache: dict[tuple, Mesh] = {}


def _floor_meshes() -> list[Mesh]:
    if not _floor_cache:
        _floor_cache.append(mesh_box((3200, 3200, 1), (0, 300, _FLOOR_MM - 0.5), MATERIAL_GROUND))
    return _floor_cache


def _board_squares(highlights: tuple) -> Mesh:
    """Squares for a white-at-bottom layout; highlights is a sorted tuple of (square, rgb)."""
    if highlights not in _board_cache:
        colored = dict(highlights)
        tiles = []
        for square in chess.SQUARES:
            file_index, rank_index = chess.square_file(square), chess.square_rank(square)
            is_light = (file_index + rank_index) % 2 == 1
            center_x = CHESS_BOARD_CENTER_MM[0] + (file_index - 3.5) * CHESS_SQUARE_MM
            center_y = CHESS_BOARD_CENTER_MM[1] + (rank_index - 3.5) * CHESS_SQUARE_MM
            tile = mesh_plate(CHESS_SQUARE_MM - 0.4, CHESS_SQUARE_MM - 0.4, (center_x, center_y, 0.0),
                            MATERIAL_BOARD_LIGHT if is_light else MATERIAL_BOARD_DARK)
            if square in colored:
                tile.face_colors[:] = np.array(colored[square]) * (1.0 if is_light else 0.55)
            tiles.append(tile)
        if len(_board_cache) > 40:
            _board_cache.clear()
        _board_cache[highlights] = mesh_merge(tiles)
    return _board_cache[highlights]


def _board_slab() -> list[Mesh]:
    half = CHESS_SQUARE_MM * 4
    return [mesh_box((2 * half + 28, 2 * half + 28, 15.6), (0, CHESS_BOARD_CENTER_MM[1], -8.2), MATERIAL_BLACK)]


def _board_frame() -> list[Mesh]:
    half = CHESS_SQUARE_MM * 4
    center_y = CHESS_BOARD_CENTER_MM[1]
    return [
        mesh_box((2 * half + 28, 14, 3), (0, center_y - half - 7, 0.6), MATERIAL_SILVER),
        mesh_box((2 * half + 28, 14, 3), (0, center_y + half + 7, 0.6), MATERIAL_SILVER),
        mesh_box((14, 2 * half, 3), (-half - 7, center_y, 0.6), MATERIAL_SILVER),
        mesh_box((14, 2 * half, 3), (half + 7, center_y, 0.6), MATERIAL_SILVER),
    ]


def world_scene_tool_position(geometry: ArmGeometry, joints_rad: np.ndarray) -> np.ndarray:
    """Tool centre point in world millimetres."""
    return arm_model_frames(geometry, joints_rad)[7][:3, 3]


@dataclass
class WorldScene:
    """The scene split by how often it changes, so the viewport can cache the still parts."""
    background: list[list[Mesh]]    # floor, board slab, board squares: always behind everything, drawn in this order
    static_objects: list[Mesh]      # rim and pieces standing still
    dynamic_objects: list[Mesh]     # arm and the piece in its gripper
    static_key: tuple               # changes whenever background or static_objects change


def world_scene_build(world: ChessWorld, geometry: ArmGeometry, joints_rad: np.ndarray, gripper_gap_mm: float,
                      highlights: dict[int, tuple]) -> WorldScene:
    """Meshes for one frame."""
    tool = world_scene_tool_position(geometry, joints_rad)
    static_objects: list[Mesh] = list(_board_frame())
    dynamic_objects: list[Mesh] = arm_visual_meshes(geometry, joints_rad, gripper_gap_mm)
    standing = []
    for piece in world.pieces.values():
        mesh = chess_pieces_mesh(piece.kind, piece.is_white)
        if piece.held:
            position = (tool[0], tool[1], tool[2] - CHESS_PIECE_GRIP_HEIGHT_MM[piece.kind] + 7.0)
            dynamic_objects.append(mesh_transform(mesh, transform_translation(*position)))
        else:
            static_objects.append(mesh_transform(mesh, transform_translation(piece.x_mm, piece.y_mm, 0.0)))
            standing.append((piece.piece_id, piece.kind, round(piece.x_mm, 1), round(piece.y_mm, 1)))
    highlight_key = tuple(sorted(highlights.items()))
    squares = _board_squares(highlight_key)
    if world.black_near_robot:  # the board is turned 180 degrees about its centre
        turn = np.array([[-1, 0, 0, 0], [0, -1, 0, 2 * CHESS_BOARD_CENTER_MM[1]], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        squares = mesh_transform(squares, turn)
    background = [_floor_meshes(), _board_slab(), [squares]]
    return WorldScene(background, static_objects, dynamic_objects,
                      (world.black_near_robot, highlight_key, tuple(sorted(standing))))
