# Chess arm simulator (branch `simulator`)

Goal: simulate a 6-joint (RRRRRR) mechanical arm playing chess; Gemini picks the robot's moves, the model is chosen at run time.

## Arm
Zero pose points straight up. J1 base yaw (Z), J2 shoulder (Y), J3 elbow (Y), J4 forearm roll (Z), J5 wrist pitch (Y), J6 tool roll (Z).
Link lengths and limits are placeholders (`core/arm_model.py`); the gripper fingers are visual only, not a 7th joint.
Inverse kinematics: damped least squares, tool pointing straight down, warm-started along straight-line paths (`core/arm_ik.py`).

## Board
Squares 40 mm, board centre at (0, 330) mm from the arm base, top at z = 0. Captured pieces go to graveyards left / right of the board.
Castling, en passant, captures and promotion move every piece physically (`core/chess_motion.py`).

## Gemini
`adapters/gemini_client.py` - REST, standard library only. Models are listed from the key (`models.list`, filtered to `generateContent`), nothing hard-coded.
Prompt = FEN + SAN history + legal UCI moves; the answer is validated with python-chess, 3 tries, then the game waits.
API key: GUI field (memory only) or `GEMINI_API_KEY`.

## Not done yet
Collision checking, real link dimensions and motor limits, speed/acceleration limits per joint, hardware bridge to the C firmware.
