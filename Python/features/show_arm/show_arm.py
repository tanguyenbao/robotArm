"""Feature: show the arm in a 3D window with one slider per joint."""
from core.arm_model import ArmGeometry, ArmPose
from ui.plot_view import plot_view_open


def show_arm_run() -> None:
    plot_view_open(ArmGeometry(), ArmPose())
