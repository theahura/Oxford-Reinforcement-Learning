"""
Author: Amol Kapoor
Description: Constants for network and other parameters.
"""
import itertools

# Slither.io params
#       Set in stone
GAME_ID = 'internet.SlitherIO-v0'
WINDOW_START = (30, 100)
WINDOW_END = (500, 370)
WINDOW_HEIGHT = WINDOW_END[1] - WINDOW_START[1]
WINDOW_WIDTH = WINDOW_END[0] - WINDOW_START[0]
#       Actions; center is about 265, 235; dont need to be specific
MOUSE_ACTIONS = [(265, 185), (290, 210), (315, 235), (290, 260), (265, 285),
                 (245, 260), (220, 235), (245, 210)]
CLICK_ACTIONS = [0, 1]
ACTIONS = list(itertools.product(MOUSE_ACTIONS, CLICK_ACTIONS))
NUM_ACTIONS = len(ACTIONS) # x coord, y coord, lmb click or not
#       Optimizations
FPS = 5.0
RESIZE_X = 128
RESIZE_Y = 200
OBSERVATION_SPACE = (RESIZE_X, RESIZE_Y, 1)

# Network Params
#       Network Constructions
CONV_LAYERS = 4
FILTER_SHAPE = [3, 3]
STRIDE = 2
OUTPUT_CHANNELS = 32
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

# Misc
STEPS_TO_SAVE = 1 # saves every n lives
ENV_STEPS = 100
NUM_WORKERS = 2
SLEEP_TIME = 300 # Wait 5 minutes before restarts

# Debugging
DEBUG = True
GLOBAL_DEBUG = True
MODEL_DEBUG = False
WORKER_DEBUG = False
RANDOM_POLICY = False
CKPT_PATH = 'data/ckpt/'
LOGDIR = 'data/logs/'
