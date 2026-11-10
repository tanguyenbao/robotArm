"""Numerical inverse kinematics (damped least squares) for the 6R arm, tool pointing straight down."""
from __future__ import annotations

import math

import numpy as np

from core.arm_model import ARM_JOINT_COUNT, ArmGeometry, arm_model_frames, arm_model_limits_rad

_ORIENTATION_WEIGHT = 150.0   # mm per radian: makes orientation errors comparable with position errors
_DAMPING_SQUARED = 4.0
_MAX_STEP_RAD = 0.35
_AXIS_COLUMN = (2, 1, 1, 2, 1, 2)  # joint axis column of its own frame: Z, Y, Y, Z, Y, Z


def arm_ik_solve(geometry: ArmGeometry, target_mm: np.ndarray, seed_joints: np.ndarray,
                 max_iterations: int = 80) -> tuple[np.ndarray, bool]:
    """Find joints that put the tool centre point on target_mm with the tool pointing down and its X axis radial.

    Returns (joints_rad, converged). Warm-start with the previous solution to stay on the same arm configuration.
    """
    limits = arm_model_limits_rad()
    azimuth = math.atan2(target_mm[1], target_mm[0])
    wanted = np.column_stack([
        (math.cos(azimuth), math.sin(azimuth), 0.0),
        (math.sin(azimuth), -math.cos(azimuth), 0.0),
        (0.0, 0.0, -1.0),
    ])  # columns: tool X (radial), tool Y (= Z cross X for a down-pointing Z), tool Z (down)
    joints = np.clip(np.asarray(seed_joints, dtype=float), limits[:, 0], limits[:, 1])
    for _ in range(max_iterations):
        frames = arm_model_frames(geometry, joints)
        tool = frames[7]
        position_error = target_mm - tool[:3, 3]
        rotation = tool[:3, :3]
        orientation_error = 0.5 * sum(np.cross(rotation[:, k], wanted[:, k]) for k in range(3))
        if np.linalg.norm(position_error) < 0.15 and np.linalg.norm(orientation_error) < 0.0015:
            return joints, True
        jacobian = np.zeros((6, ARM_JOINT_COUNT))
        for index in range(ARM_JOINT_COUNT):
            axis = frames[index + 1][:3, _AXIS_COLUMN[index]]
            origin = frames[index + 1][:3, 3]
            jacobian[:3, index] = np.cross(axis, tool[:3, 3] - origin)
            jacobian[3:, index] = axis * _ORIENTATION_WEIGHT
        error = np.concatenate([position_error, orientation_error * _ORIENTATION_WEIGHT])
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + _DAMPING_SQUARED * np.eye(6), error)
        largest = np.max(np.abs(step))
        if largest > _MAX_STEP_RAD:
            step *= _MAX_STEP_RAD / largest
        joints = np.clip(joints + step, limits[:, 0], limits[:, 1])
    return joints, False
