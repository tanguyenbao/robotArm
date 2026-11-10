"""Quad meshes with per-face colour and material (pure numpy).

Every face is a quad (4 vertex indices). A triangle repeats its last index. Winding is counter-clockwise seen from outside.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MATERIAL_SILVER = 0
MATERIAL_BLACK = 1
MATERIAL_BOARD_LIGHT = 2
MATERIAL_BOARD_DARK = 3
MATERIAL_GROUND = 4


@dataclass(frozen=True)
class MaterialSpec:
    base_color: tuple[float, float, float]
    metal: float          # 0 = matte, 1 = mirror-like
    reflect_white: float  # how much white the reflections add (black metal mirrors a grey room)
    specular: float
    shininess: float


MATERIAL_TABLE = (
    MaterialSpec((0.80, 0.82, 0.86), 0.80, 0.00, 0.90, 60.0),   # silver metallic
    MaterialSpec((0.055, 0.060, 0.070), 0.75, 0.40, 0.85, 70.0),  # black metallic
    MaterialSpec((0.80, 0.82, 0.86), 0.12, 0.00, 0.25, 20.0),   # board, light square
    MaterialSpec((0.16, 0.18, 0.22), 0.12, 0.00, 0.25, 20.0),   # board, dark square
    MaterialSpec((0.12, 0.135, 0.16), 0.00, 0.00, 0.00, 1.0),   # floor
)


@dataclass
class Mesh:
    vertices: np.ndarray       # (N, 3) float
    faces: np.ndarray          # (M, 4) int
    face_colors: np.ndarray    # (M, 3) float 0..1
    face_materials: np.ndarray  # (M,) int


def _mesh_build(vertices: np.ndarray, faces: list | np.ndarray, material: int, color: tuple | None) -> Mesh:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    first, second, third, fourth = (vertices[faces[:, k]] for k in range(4))
    area = np.linalg.norm(np.cross(third - first, fourth - second), axis=1)
    faces = faces[area > 1e-6]
    base = np.array(color if color is not None else MATERIAL_TABLE[material].base_color, dtype=float)
    return Mesh(vertices, faces, np.tile(base, (len(faces), 1)), np.full(len(faces), material, dtype=int))


def mesh_lathe(profile: list[tuple[float, float]], segments: int = 24, material: int = MATERIAL_SILVER,
               color: tuple | None = None) -> Mesh:
    """Revolve a (radius, z) profile, listed bottom to top, around the Z axis."""
    profile_points = np.asarray(profile, dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cosines, sines = np.cos(angles), np.sin(angles)
    rings = [np.column_stack([radius * cosines, radius * sines, np.full(segments, height)]) for radius, height in profile_points]
    vertices = np.vstack(rings)
    faces = []
    for ring in range(len(profile_points) - 1):
        for index in range(segments):
            following = (index + 1) % segments
            faces.append((ring * segments + index, ring * segments + following,
                          (ring + 1) * segments + following, (ring + 1) * segments + index))
    return _mesh_build(vertices, faces, material, color)


def mesh_cylinder(radius: float, length: float, axis: str = "z", center: tuple = (0.0, 0.0, 0.0),
                  material: int = MATERIAL_SILVER, color: tuple | None = None, segments: int = 24,
                  radius_top: float | None = None) -> Mesh:
    """Capped cylinder (or cone frustum) along x, y or z, centred on `center`."""
    top = radius if radius_top is None else radius_top
    mesh = mesh_lathe([(0.0, -length / 2), (radius, -length / 2), (top, length / 2), (0.0, length / 2)], segments, material, color)
    if axis == "x":
        mesh.vertices = mesh.vertices[:, [2, 0, 1]]
    elif axis == "y":
        mesh.vertices = mesh.vertices[:, [1, 2, 0]]
    mesh.vertices = mesh.vertices + np.asarray(center, dtype=float)
    return mesh


def mesh_box(size: tuple[float, float, float], center: tuple = (0.0, 0.0, 0.0), material: int = MATERIAL_SILVER,
             color: tuple | None = None) -> Mesh:
    """Axis-aligned box."""
    half_x, half_y, half_z = (value / 2 for value in size)
    corners = np.array([
        [-half_x, -half_y, -half_z], [half_x, -half_y, -half_z], [half_x, half_y, -half_z], [-half_x, half_y, -half_z],
        [-half_x, -half_y, half_z], [half_x, -half_y, half_z], [half_x, half_y, half_z], [-half_x, half_y, half_z],
    ]) + np.asarray(center, dtype=float)
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    return _mesh_build(corners, faces, material, color)


def mesh_plate(size_x: float, size_y: float, center: tuple, material: int = MATERIAL_SILVER,
               color: tuple | None = None) -> Mesh:
    """One flat quad facing +Z (a tile)."""
    half_x, half_y = size_x / 2, size_y / 2
    corners = np.array([[-half_x, -half_y, 0.0], [half_x, -half_y, 0.0], [half_x, half_y, 0.0], [-half_x, half_y, 0.0]]) + np.asarray(center, dtype=float)
    return _mesh_build(corners, [(0, 1, 2, 3)], material, color)


def mesh_transform(mesh: Mesh, matrix: np.ndarray) -> Mesh:
    """Return a copy of the mesh moved by a 4x4 matrix (colours and faces are shared)."""
    moved = mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3]
    return Mesh(moved, mesh.faces, mesh.face_colors, mesh.face_materials)


def mesh_merge(meshes: list[Mesh]) -> Mesh:
    """Join meshes into one."""
    offsets = np.cumsum([0] + [len(mesh.vertices) for mesh in meshes[:-1]])
    return Mesh(
        np.vstack([mesh.vertices for mesh in meshes]),
        np.vstack([mesh.faces + offset for mesh, offset in zip(meshes, offsets)]),
        np.vstack([mesh.face_colors for mesh in meshes]),
        np.concatenate([mesh.face_materials for mesh in meshes]),
    )
