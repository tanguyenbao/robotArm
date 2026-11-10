"""6-DOF all-revolute arm (R-R-R-R-R-R): forward kinematics and joint limits.

Zero pose: every link points straight up (+Z). Joint axes in the local frame after the previous joints:
J1 base yaw (Z), J2 shoulder (Y), J3 elbow (Y), J4 forearm roll (Z), J5 wrist pitch (Y), J6 tool roll (Z).
Tool pointing straight down means J2 + J3 + J5 = 180 degrees with J4 = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.transforms import transform_rotation_y, transform_rotation_z, transform_translation

ARM_JOINT_COUNT = 6
ARM_JOINT_NAMES = ("Base", "Shoulder", "Elbow", "Forearm roll", "Wrist", "Tool roll")
ARM_JOINT_LIMITS_DEG = np.array([[-170, 170], [-100, 100], [-155, 155], [-170, 170], [-125, 125], [-180, 180]], dtype=float)
ARM_HOME_DEG = (0.0, 40.0, 70.0, 0.0, 70.0, 0.0)


@dataclass(frozen=True)
class ArmGeometry:
    shoulder_height_mm: float = 130.0
    upper_arm_mm: float = 290.0
    forearm_mm: float = 250.0
    wrist_offset_mm: float = 60.0
    wrist_to_flange_mm: float = 50.0
    tool_length_mm: float = 110.0


def arm_model_home_joints() -> np.ndarray:
    """Home pose in radians (tool pointing down)."""
    return np.radians(ARM_HOME_DEG)


def arm_model_limits_rad() -> np.ndarray:
    """(6, 2) joint limits in radians."""
    return np.radians(ARM_JOINT_LIMITS_DEG)


def arm_model_frames(geometry: ArmGeometry, joints_rad: np.ndarray) -> np.ndarray:
    """Return (8, 4, 4): base, frames after J1..J6, and the tool centre point (fingertip line)."""
    frames = np.empty((8, 4, 4))
    frames[0] = np.eye(4)
    frames[1] = transform_rotation_z(joints_rad[0])
    frames[2] = frames[1] @ transform_translation(0, 0, geometry.shoulder_height_mm) @ transform_rotation_y(joints_rad[1])
    frames[3] = frames[2] @ transform_translation(0, 0, geometry.upper_arm_mm) @ transform_rotation_y(joints_rad[2])
    frames[4] = frames[3] @ transform_translation(0, 0, geometry.forearm_mm) @ transform_rotation_z(joints_rad[3])
    frames[5] = frames[4] @ transform_translation(0, 0, geometry.wrist_offset_mm) @ transform_rotation_y(joints_rad[4])
    frames[6] = frames[5] @ transform_translation(0, 0, geometry.wrist_to_flange_mm) @ transform_rotation_z(joints_rad[5])
    frames[7] = frames[6] @ transform_translation(0, 0, geometry.tool_length_mm)
    return frames
