"""
Author: Amol Kapoor
Description: Constants for network and other parameters.
"""

# Slither.io params
#       Set in stone
GAME_ID = 'internet.SlitherIO-v0'
WINDOW_START = (18, 84)
WINDOW_END = (521, 384)
WINDOW_HEIGHT = WINDOW_END[1] - WINDOW_START[1]
WINDOW_WIDTH = WINDOW_END[0] - WINDOW_START[0]
NUM_ACTIONS = 3 # x coord, y coord, lmb click or not
#       Optimizations
FPS = 5.0
RESIZE_X = 128
RESIZE_Y = 200

# Network Params
#       Network Constructions
CONV_LAYERS = 4
FILTER_SHAPE = [3, 3]
STRIDE = 2
LSTM_UNITS = 256

#       Optimizations and Convergence Tricks
INPUT_KEEP_PROB = .5
OUTPUT_KEEP_PROB = .5

# Debugging
DEBUG = False
RANDOM_POLICY = False
