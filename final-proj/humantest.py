import sys
import select
import tty
import termios

old_settings = termios.tcgetattr(sys.stdin)

def isData():
    """
    Checks if there is anything in stdin
    """
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def convert_to_action(ch):
    """
    Converts from wasd to action arrays
    """
    if ch == 'a':
        return [0, 0, 0, 1, 0, 0]
    elif ch == 'd':
        return [0, 0, 1, 0, 0, 0]
    elif ch == 'w':
        return [0, 1, 0, 0, 0, 0]
    elif ch == 'q':
        return [0, 0, 0, 0, 1, 0]
    elif ch == 'e':
        return [0, 0, 0, 0, 0, 1]
    elif ch == 's':
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
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
