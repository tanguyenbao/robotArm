"""4x4 homogeneous transform helpers (pure numpy)."""
from __future__ import annotations

import math

import numpy as np


def transform_translation(x_mm: float, y_mm: float, z_mm: float) -> np.ndarray:
    """Translation matrix."""
    matrix = np.eye(4)
    matrix[:3, 3] = (x_mm, y_mm, z_mm)
    return matrix


def transform_rotation_x(angle_rad: float) -> np.ndarray:
    """Rotation about the X axis."""
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    matrix = np.eye(4)
    matrix[1, 1], matrix[1, 2], matrix[2, 1], matrix[2, 2] = cosine, -sine, sine, cosine
    return matrix


def transform_rotation_y(angle_rad: float) -> np.ndarray:
    """Rotation about the Y axis (positive angle tips +Z toward +X)."""
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    matrix = np.eye(4)
    matrix[0, 0], matrix[0, 2], matrix[2, 0], matrix[2, 2] = cosine, sine, -sine, cosine
    return matrix


def transform_rotation_z(angle_rad: float) -> np.ndarray:
    """Rotation about the Z axis."""
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    matrix = np.eye(4)
    matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1] = cosine, -sine, sine, cosine
    return matrix
