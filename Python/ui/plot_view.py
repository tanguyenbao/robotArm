"""Matplotlib 3D view of the arm with joint sliders."""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from core.arm_model import ArmGeometry, ArmPose, arm_model_joint_positions


def plot_view_open(geometry: ArmGeometry, pose: ArmPose) -> None:
    figure = plt.figure("RobotArm simulator")
    axes = figure.add_axes([0.05, 0.25, 0.9, 0.7], projection="3d")
    reach_mm = geometry.upper_arm_mm + geometry.forearm_mm

    def redraw() -> None:
        points = arm_model_joint_positions(geometry, pose)
        axes.clear()
        axes.plot(points[:, 0], points[:, 1], points[:, 2], "o-", linewidth=4)
        axes.set_xlim(-reach_mm, reach_mm)
        axes.set_ylim(-reach_mm, reach_mm)
        axes.set_zlim(0, reach_mm + geometry.base_height_mm)
        axes.set_xlabel("X (mm)")
        axes.set_ylabel("Y (mm)")
        axes.set_zlabel("Z (mm)")
        figure.canvas.draw_idle()

    def add_slider(vertical_position: float, label: str, low: float, high: float, start: float, attribute: str) -> Slider:
        slider = Slider(figure.add_axes([0.2, vertical_position, 0.65, 0.03]), label, low, high, valinit=start)

        def on_change(value: float) -> None:
            setattr(pose, attribute, value)
            redraw()

        slider.on_changed(on_change)
        return slider

    sliders = [
        add_slider(0.15, "Base yaw", -180, 180, pose.base_yaw_deg, "base_yaw_deg"),
        add_slider(0.10, "Shoulder", -90, 90, pose.shoulder_pitch_deg, "shoulder_pitch_deg"),
        add_slider(0.05, "Elbow", -150, 150, pose.elbow_pitch_deg, "elbow_pitch_deg"),
    ]
    redraw()
    plt.show()
    del sliders
