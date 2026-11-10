# RobotArm

Robot arm project. Layout follows the ElderScroll project.

| Folder | Purpose |
|---|---|
| `Python/` | PC side: simulator, kinematics, control software (layers: core / adapters / features / ui) |
| `C/` | Firmware for the arm controller (layers: 0_hal / 1_driver / 2_protocol / 3_service / 4_app + board) |
| `Tool/` | PC helper tools (tools hub, test and bring-up scripts) |
| `.Doc/` | Plans, notes, datasheets, test reports |
| `.agent/` | Agent / handoff rules and working files |

## Branches
- `main` - project skeleton.
- `simulator` - 6R robot arm playing chess against Gemini, 3D simulator (Python). Start with `Tool/run_gui.ps1`.

## Editor setup
- Python: create `.venv` in the project root and select it as the VS Code interpreter.
- C: needs `compile_commands.json` from the user's build; point clangd / IntelliSense at it.
