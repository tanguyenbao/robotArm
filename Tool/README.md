# Tool

PC helper tools for the robot arm.

| File | Purpose |
|---|---|
| `run_gui.ps1` | Creates `.venv` in the project root when missing, installs `Python/requirements.txt` when a library is missing, starts the chess arm simulator. `-Setup` reinstalls, `-SetupOnly` stops after setup. |

Run:  `powershell -ExecutionPolicy Bypass -File Tool\run_gui.ps1`
