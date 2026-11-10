# Python

PC-side software for the robot arm.

Layers (bottom to top): `core/` (pure logic) -> `adapters/` (outside world) -> `features/` (one feature each, with `help.txt`, listed in `features.json`) -> `ui/` + `main.py`.

Empty on `main`; the simulator lives on the `simulator` branch.
