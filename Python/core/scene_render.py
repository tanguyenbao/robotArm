"""Small software 3D renderer: perspective camera, back-face culling, metallic shading, depth sort. Pure numpy."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.mesh import MATERIAL_TABLE, Mesh, mesh_merge

_NEAR_MM = 20.0
_KEY_LIGHT = np.array([-0.45, -0.35, 0.82])
_FILL_LIGHT = np.array([0.7, 0.5, 0.35])
_METAL = np.array([spec.metal for spec in MATERIAL_TABLE])
_REFLECT_WHITE = np.array([spec.reflect_white for spec in MATERIAL_TABLE])
_SPECULAR = np.array([spec.specular for spec in MATERIAL_TABLE])
_SHININESS = np.array([spec.shininess for spec in MATERIAL_TABLE])


@dataclass
class SceneCamera:
    azimuth_deg: float = -38.0
    elevation_deg: float = 34.0
    distance_mm: float = 1120.0
    target_mm: np.ndarray = field(default_factory=lambda: np.array([0.0, 270.0, 90.0]))
    fov_deg: float = 34.0


@dataclass
class FaceBatch:
    """Visible faces of some meshes, not sorted yet."""
    points: np.ndarray   # (K, 4, 2) screen coordinates
    colors: np.ndarray   # (K, 3) uint8
    depth: np.ndarray    # (K,) distance from the camera, larger = farther


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-9)


def _shade(face_colors: np.ndarray, materials: np.ndarray, normals: np.ndarray, centers: np.ndarray,
           eye: np.ndarray) -> np.ndarray:
    view = _unit(eye - centers)
    key, fill = _unit(_KEY_LIGHT), _unit(_FILL_LIGHT)

    diffuse = np.clip(normals @ key, 0, 1) * 0.8 + np.clip(normals @ fill, 0, 1) * 0.25
    half = _unit(key + view)
    specular = np.clip(np.sum(normals * half, axis=1), 0, 1) ** _SHININESS[materials] * _SPECULAR[materials]

    reflected = 2.0 * np.sum(normals * view, axis=1, keepdims=True) * normals - view
    sky = np.clip((reflected[:, 2] + 0.10) / 1.10, 0, 1) ** 0.8
    softbox = np.exp(-(((reflected[:, 2] - 0.42) / 0.10) ** 2)) * 0.45 * (0.6 + 0.4 * reflected[:, 0])
    horizon = np.exp(-((reflected[:, 2] / 0.07) ** 2)) * -0.28
    toward_key = np.clip(np.sum(reflected * key, axis=1), 0, 1) ** 2 * 0.35
    environment = np.clip(0.40 + 0.55 * sky + softbox + horizon + toward_key, 0.12, 1.25)

    matte = face_colors * (0.22 + diffuse)[:, None]
    white = _REFLECT_WHITE[materials][:, None]
    tint = face_colors * (1.0 - white) + white
    mirror = tint * environment[:, None] * 1.05 * (0.55 + 0.45 * diffuse)[:, None]
    metal = _METAL[materials][:, None]
    color = matte * (1.0 - metal) + mirror * metal + specular[:, None]
    return np.clip(color, 0.0, 1.0)


def _camera_setup(camera: SceneCamera, height: int) -> tuple[np.ndarray, tuple, float]:
    azimuth, elevation = math.radians(camera.azimuth_deg), math.radians(camera.elevation_deg)
    offset = np.array([math.sin(azimuth) * math.cos(elevation), -math.cos(azimuth) * math.cos(elevation), math.sin(elevation)])
    eye = np.asarray(camera.target_mm, dtype=float) + camera.distance_mm * offset
    forward = -offset
    right = _unit(np.cross(forward, np.array([0.0, 0.0, 1.0])))
    up = np.cross(right, forward)
    focal = (height / 2) / math.tan(math.radians(camera.fov_deg) / 2)
    return eye, (right, up, forward), focal


def scene_render_faces(meshes: list[Mesh], camera: SceneCamera, width: int, height: int) -> FaceBatch:
    """Project and shade the visible faces of these meshes."""
    empty = FaceBatch(np.zeros((0, 4, 2)), np.zeros((0, 3), dtype=np.uint8), np.zeros(0))
    if not meshes:
        return empty
    eye, (right, up, forward), focal = _camera_setup(camera, height)
    mesh = mesh_merge(meshes)
    relative = mesh.vertices - eye
    camera_space = np.column_stack([relative @ right, relative @ up, relative @ forward])
    depth = np.maximum(camera_space[:, 2], 1e-6)
    screen = np.column_stack([width / 2 + focal * camera_space[:, 0] / depth, height / 2 - focal * camera_space[:, 1] / depth])

    corners = mesh.vertices[mesh.faces]                       # (M, 4, 3)
    normals = _unit(np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1]))
    centers = corners.mean(axis=1)
    visible = np.sum(normals * (eye - centers), axis=1) > 0
    visible &= np.all(camera_space[mesh.faces, 2] > _NEAR_MM, axis=1)
    kept = np.nonzero(visible)[0]
    if len(kept) == 0:
        return empty
    colors = _shade(mesh.face_colors[kept], mesh.face_materials[kept], normals[kept], centers[kept], eye)
    return FaceBatch(screen[mesh.faces[kept]], (colors * 255).astype(np.uint8), camera_space[mesh.faces[kept], 2].mean(axis=1))


def scene_render_order(depth: np.ndarray) -> np.ndarray:
    """Indices that draw far faces first (painter's algorithm)."""
    return np.argsort(-depth, kind="stable")


def scene_render_batch_join(first: FaceBatch, second: FaceBatch) -> FaceBatch:
    """Concatenate two batches (faces of `first` come first in the index space)."""
    return FaceBatch(np.concatenate([first.points, second.points]), np.concatenate([first.colors, second.colors]),
                     np.concatenate([first.depth, second.depth]))
