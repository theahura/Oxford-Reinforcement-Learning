import sys
import select
import tty
import termios

old_settings = termios.tcgetattr(sys.stdin)
toggle = False

def isData():
    """
    Checks if there is anything in stdin
    """
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def convert_to_action(ch):
    """
    Converts from wasd to action arrays
    """
    global toggle
    if toggle:
        if ch == 'a':
            return [0, 0, 0, 0, 1, 0]
        elif ch == 'd':
            return [0, 0, 0, 0, 0, 1]
        elif ch == 's':
            toggle = False
            return [1, 0, 0, 0, 0, 0]
        else:
            return [0, 1, 0, 0, 0, 0]
    else:
        if ch == 'a':
            return [0, 0, 0, 1, 0, 0]
        elif ch == 'd':
            return [0, 0, 1, 0, 0, 0]
        elif ch == 'w':
            toggle = True
            return [0, 1, 0, 0, 0, 0]
        else:
            return [1, 0, 0, 0, 0, 0]

def setup_keyboard():
    """
    Sets up keyboard for input to universe.
    """
    tty.setcbreak(sys.stdin.fileno())

def return_keyboard():
    """
    Returns keyboard to old settings.
    """
    global old_settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
