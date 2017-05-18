"""
Author: Amol Kapoor
Description: Constants for network and other parameters.
"""

# Slither.io params
#       Set in stone
GAME_ID = 'internet.SlitherIO-v0'
WINDOW_START = (30, 100)
WINDOW_END = (500, 370)
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
#       A3C Params
VF_LOSS_CONST = 0.5
ENT_CONST = 0.01
LEARNING_RATE = 1e-4
GAMMA = 0.99
LAMBDA = 1.0
#       Optimizations and Convergence Tricks
INPUT_KEEP_PROB = .5
OUTPUT_KEEP_PROB = .5
MAX_GRAD_NORM = 40.0

# Debugging
DEBUG = False
RANDOM_POLICY = False
CKPT_PATH = 'data/ckpt/'
DEBUG_STEPS = 10
