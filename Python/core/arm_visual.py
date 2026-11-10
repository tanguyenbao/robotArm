"""Mechanical 3D look of the arm: one mesh group per link, built in that link's own frame."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from core.arm_model import ArmGeometry, arm_model_frames
from core.mesh import (MATERIAL_BLACK, MATERIAL_SILVER, Mesh, mesh_box, mesh_cylinder, mesh_lathe, mesh_merge,
                       mesh_transform)
from core.transforms import transform_translation

ARM_GRIPPER_OPEN_MM = 34.0
ARM_GRIPPER_PAD_LENGTH_MM = 14.0
_FINGER_WIDTH_MM = 8.0


def _link_base() -> Mesh:
    return mesh_merge([
        mesh_box((250, 250, 16), (0, 0, -8), MATERIAL_BLACK),
        mesh_lathe([(0, 0), (100, 0), (100, 16), (90, 26), (0, 26)], 32, MATERIAL_BLACK),
        mesh_lathe([(0, 26), (78, 26), (78, 50), (70, 58), (0, 58)], 32, MATERIAL_SILVER),
    ])


def _link_turret(geometry: ArmGeometry) -> Mesh:
    shoulder = geometry.shoulder_height_mm
    return mesh_merge([
        mesh_lathe([(0, 58), (66, 58), (66, 66), (60, 70), (60, 96), (0, 96)], 32, MATERIAL_SILVER),
        mesh_cylinder(67, 8, "z", (0, 0, 80), MATERIAL_BLACK, segments=32),
        mesh_box((84, 124, 46), (0, 0, shoulder - 28), MATERIAL_SILVER),
        mesh_box((78, 18, 96), (0, -53, shoulder + 4), MATERIAL_SILVER),
        mesh_box((78, 18, 96), (0, 53, shoulder + 4), MATERIAL_SILVER),
        mesh_cylinder(34, 140, "y", (0, 0, shoulder), MATERIAL_BLACK),
        mesh_cylinder(25, 6, "y", (0, -73, shoulder), MATERIAL_SILVER),
        mesh_cylinder(25, 6, "y", (0, 73, shoulder), MATERIAL_SILVER),
    ])


def _link_upper_arm(geometry: ArmGeometry) -> Mesh:
    length = geometry.upper_arm_mm
    return mesh_merge([
        mesh_cylinder(38, 80, "y", (0, 0, 0), MATERIAL_SILVER),
        mesh_box((56, 60, length - 70), (0, 0, length / 2), MATERIAL_SILVER),
        mesh_box((60, 24, length - 150), (0, 0, length / 2), MATERIAL_BLACK),
        mesh_cylinder(40, 92, "y", (0, 0, length), MATERIAL_BLACK),
        mesh_cylinder(28, 6, "y", (0, -49, length), MATERIAL_SILVER),
        mesh_cylinder(28, 6, "y", (0, 49, length), MATERIAL_SILVER),
    ])


def _link_forearm(geometry: ArmGeometry) -> Mesh:
    length = geometry.forearm_mm
    return mesh_merge([
        mesh_cylinder(34, 70, "y", (0, 0, 0), MATERIAL_SILVER),
        mesh_box((50, 54, length - 80), (0, 0, (length - 80) / 2 + 28), MATERIAL_SILVER),
        mesh_box((54, 18, length - 150), (0, 0, length / 2), MATERIAL_BLACK),
        mesh_cylinder(34, 40, "z", (0, 0, length - 20), MATERIAL_BLACK),
    ])


def _link_wrist_roll(geometry: ArmGeometry) -> Mesh:
    offset = geometry.wrist_offset_mm
    return mesh_merge([
        mesh_cylinder(30, offset - 18, "z", (0, 0, (offset - 18) / 2), MATERIAL_SILVER),
        mesh_box((44, 70, 34), (0, 0, offset - 18), MATERIAL_SILVER),
        mesh_cylinder(26, 80, "y", (0, 0, offset), MATERIAL_BLACK),
        mesh_cylinder(18, 6, "y", (0, -43, offset), MATERIAL_SILVER),
        mesh_cylinder(18, 6, "y", (0, 43, offset), MATERIAL_SILVER),
    ])


def _link_wrist_pitch(geometry: ArmGeometry) -> Mesh:
    length = geometry.wrist_to_flange_mm
    return mesh_merge([
        mesh_cylinder(22, 56, "y", (0, 0, 0), MATERIAL_SILVER),
        mesh_box((36, 40, length - 14), (0, 0, (length - 14) / 2 + 8), MATERIAL_SILVER),
    ])


def _link_gripper_body(geometry: ArmGeometry) -> Mesh:
    tool = geometry.tool_length_mm
    return mesh_merge([
        mesh_cylinder(30, 12, "z", (0, 0, -6), MATERIAL_BLACK),
        mesh_box((64, 36, 34), (0, 0, 17), MATERIAL_BLACK),
        mesh_box((92, 16, 8), (0, 0, 38), MATERIAL_SILVER),
        mesh_box((60, 20, 4), (0, 0, tool - 40), MATERIAL_BLACK),
    ])


@lru_cache(maxsize=4)
def _static_links(geometry: ArmGeometry) -> tuple[Mesh, ...]:
    return (_link_base(), _link_turret(geometry), _link_upper_arm(geometry), _link_forearm(geometry),
            _link_wrist_roll(geometry), _link_wrist_pitch(geometry), _link_gripper_body(geometry))


@lru_cache(maxsize=4)
def _finger_prototypes(geometry: ArmGeometry) -> tuple[Mesh, Mesh]:
    tool = geometry.tool_length_mm
    finger_length = tool - 42
    finger = mesh_merge([
        mesh_box((_FINGER_WIDTH_MM, 26, finger_length), (0, 0, 42 + finger_length / 2), MATERIAL_SILVER),
        mesh_box((3, 22, ARM_GRIPPER_PAD_LENGTH_MM), (_FINGER_WIDTH_MM / 2 + 1.5, 0, tool - ARM_GRIPPER_PAD_LENGTH_MM / 2), MATERIAL_BLACK),
    ])
    mirrored = mesh_merge([
        mesh_box((_FINGER_WIDTH_MM, 26, finger_length), (0, 0, 42 + finger_length / 2), MATERIAL_SILVER),
        mesh_box((3, 22, ARM_GRIPPER_PAD_LENGTH_MM), (-_FINGER_WIDTH_MM / 2 - 1.5, 0, tool - ARM_GRIPPER_PAD_LENGTH_MM / 2), MATERIAL_BLACK),
    ])
    return finger, mirrored


def arm_visual_meshes(geometry: ArmGeometry, joints_rad: np.ndarray, gripper_gap_mm: float) -> list[Mesh]:
    """World-space meshes of the whole arm for the given joints and gripper opening."""
    frames = arm_model_frames(geometry, joints_rad)
    static = _static_links(geometry)
    link_frames = (0, 1, 2, 3, 4, 5, 6)
    meshes = [mesh_transform(link, frames[frame]) for link, frame in zip(static, link_frames)]
    left, right = _finger_prototypes(geometry)
    offset = gripper_gap_mm / 2 + _FINGER_WIDTH_MM / 2
    meshes.append(mesh_transform(left, frames[6] @ transform_translation(-offset, 0, 0)))
    meshes.append(mesh_transform(right, frames[6] @ transform_translation(offset, 0, 0)))
    return meshes
