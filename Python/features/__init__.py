from features.chess_arm.chess_arm_controller import ChessArmController

# feature name -> controller class; main.py hands the class to the UI
FEATURE_TABLE = {
    "chess_arm": ChessArmController,
}
