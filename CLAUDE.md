# RobotArm - project rules

- The global rules of the user (`~/.claude/CLAUDE.md`) apply: never build, layered architecture, a `help.txt` per feature, Python only in the project `.venv`.
- Python layers: `core` -> `adapters` -> `features` -> `ui` + `main.py`. C layers: `0_hal` -> `1_driver` -> `2_protocol` -> `3_service` -> `4_app`, plus constants-only `board/`.
- Hardware facts (motors, pins, limits) come from the user. Do not guess them.
- Plans and notes go in `.Doc/`.
