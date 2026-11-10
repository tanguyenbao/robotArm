# Python

PC-side software for the robot arm. Branch `simulator`: a 3-DOF arm (base yaw, shoulder, elbow) shown in 3D with sliders.

Layers (bottom to top): `core/` (arm model, forward kinematics) -> `adapters/` (empty) -> `features/` (`show_arm`, with `help.txt`, listed in `features.json`) -> `ui/` + `main.py`.

Run (from `Python/`, inside the project `.venv`):
```
pip install -r requirements.txt
python main.py
```
Link lengths and joint limits are placeholders until the real arm is decided.
