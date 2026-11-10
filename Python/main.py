"""Entrypoint of the RobotArm simulator. Usage: python main.py [feature]   (default feature: chess_arm)"""
import sys

from features import FEATURE_TABLE
from ui.main_window import main_window_run


def main() -> None:
    feature_name = sys.argv[1] if len(sys.argv) > 1 else "chess_arm"
    if feature_name not in FEATURE_TABLE:
        print(f"Unknown feature '{feature_name}'. Available: {', '.join(FEATURE_TABLE)}")
        sys.exit(1)
    main_window_run(FEATURE_TABLE[feature_name])


if __name__ == "__main__":
    main()
