# Python

PC-side software for the robot arm. Branch `simulator`: a 6-joint (RRRRRR) arm plays chess or xiangqi in a 3D view; a built-in engine chooses the robot's moves and Gemini Live is the chatty companion (voice or text).

Layers (bottom to top):
- `core/` - arm model, kinematics, meshes, 3D renderer, chess / xiangqi worlds, rules, engines, motion planning, companion prompt, audio conversion. No UI, no network.
- `adapters/` - `gemini_client.py` (Gemini REST, model list), `gemini_live_session.py` (Live WebSocket conversation), `audio_io.py` (speaker and microphone).
- `features/` - `chess_arm` (game + arm animation + engine player + Gemini companion), with `help.txt`, listed in `features.json`.
- `ui/` + `main.py` - glass window, metal buttons, 3D viewport.

Run (from the project root):
```
powershell -ExecutionPolicy Bypass -File Tool\run_gui.ps1
```
or inside the project `.venv`: `pip install -r Python/requirements.txt`, then `python Python/main.py`.

Companion: press **Refresh models** after typing the API key (or set `GEMINI_API_KEY`), pick a Live model, press **Connect companion**, then play (Mic button to talk). See `features/chess_arm/help.txt`.
