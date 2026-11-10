# Python

PC-side software for the robot arm. Branch `simulator`: a 6-joint (RRRRRR) arm plays chess in a 3D view; Gemini chooses the robot's moves.

Layers (bottom to top):
- `core/` - arm model, kinematics, meshes, 3D renderer, chess world, motion planning, Gemini prompt. No UI, no network.
- `adapters/` - `gemini_client.py` (Gemini REST).
- `features/` - `chess_arm` (game + arm animation + Gemini player), with `help.txt`, listed in `features.json`.
- `ui/` + `main.py` - glass window, metal buttons, 3D viewport.

Run (from the project root):
```
powershell -ExecutionPolicy Bypass -File Tool\run_gui.ps1
```
or inside the project `.venv`: `pip install -r Python/requirements.txt`, then `python Python/main.py`.

Gemini: press **Refresh models** after typing the API key (or set `GEMINI_API_KEY`), pick a model, then play. See `features/chess_arm/help.txt`.
