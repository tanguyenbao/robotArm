"""Pure robot arm model: 3-DOF arm (base yaw, shoulder pitch, elbow pitch) with forward kinematics.

Link lengths are placeholders until the real arm is decided.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ArmGeometry:
    base_height_mm: float = 80.0
    upper_arm_mm: float = 120.0
    forearm_mm: float = 100.0


@dataclass
class ArmPose:
    base_yaw_deg: float = 0.0
    shoulder_pitch_deg: float = 45.0
    elbow_pitch_deg: float = -60.0


def arm_model_joint_positions(geometry: ArmGeometry, pose: ArmPose) -> np.ndarray:
    """Return a (4, 3) array: base, shoulder, elbow, tool tip positions in mm."""
    yaw = math.radians(pose.base_yaw_deg)
    shoulder = math.radians(pose.shoulder_pitch_deg)
    elbow_total = shoulder + math.radians(pose.elbow_pitch_deg)

    base = np.array([0.0, 0.0, 0.0])
    shoulder_point = np.array([0.0, 0.0, geometry.base_height_mm])

    def planar_to_world(radial_mm: float, height_mm: float) -> np.ndarray:
        return np.array([radial_mm * math.cos(yaw), radial_mm * math.sin(yaw), height_mm])

    elbow_point = shoulder_point + planar_to_world(
        geometry.upper_arm_mm * math.cos(shoulder), geometry.upper_arm_mm * math.sin(shoulder)
    )
    tip_point = elbow_point + planar_to_world(
        geometry.forearm_mm * math.cos(elbow_total), geometry.forearm_mm * math.sin(elbow_total)
    )
    return np.vstack([base, shoulder_point, elbow_point, tip_point])
