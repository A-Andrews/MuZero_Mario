import numpy as np

N_ACTIONS = 12


def complex_movement_to_button_presses(action):
    """Convert an action in COMPLEX_MOVEMENT moveset to a binary vector of button presses.

    Buttons: [run, up, down, left, right, jump]
    COMPLEX_MOVEMENT (A=jump, B=run):
        0 NOOP, 1 right, 2 right+A, 3 right+B, 4 right+A+B,
        5 A, 6 left, 7 left+A, 8 left+B, 9 left+A+B,
        10 down, 11 up
    """
    buttons = np.zeros(6, dtype=bool)
    if action in (1, 2, 3, 4):
        buttons[4] = True  # right
    if action in (6, 7, 8, 9):
        buttons[3] = True  # left
    if action in (2, 4, 5, 7, 9):
        buttons[5] = True  # jump
    if action in (3, 4, 8, 9):
        buttons[0] = True  # run
    if action == 10:
        buttons[2] = True  # down
    if action == 11:
        buttons[1] = True  # up
    return buttons
