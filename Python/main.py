"""Entrypoint of the RobotArm simulator. Usage: python main.py [feature]"""
import sys

from features import FEATURE_TABLE


def main() -> None:
    feature_name = sys.argv[1] if len(sys.argv) > 1 else "show_arm"
    if feature_name not in FEATURE_TABLE:
        print(f"Unknown feature '{feature_name}'. Available: {', '.join(FEATURE_TABLE)}")
        sys.exit(1)
    FEATURE_TABLE[feature_name]()


if __name__ == "__main__":
    main()
