import numpy as np

""" GAME OPTIONS """
total_players = 2
epochs = 5
fps = 25
game_length = fps * 20
display_frame = True
use_grid = False
players = [
    {
        "feedforward": True,
        "random": False,
        "hidden_size": 50,
    },
    {
        "feedforward": True,
        "random": False,
        "hidden_size": 50,
    }
]
""" END GAME OPTIONS"""

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 800
wall_offset = 100
wall_width = 20
GAME_WIDTH = SCREEN_WIDTH - (wall_offset + wall_width) * 2
GAME_HEIGHT = SCREEN_HEIGHT - (wall_offset + wall_width) * 2
EXTRA_LAYERS = 1
DATA_PER_PLAYER = 4
collision_types = {'player': 1, 'bullet': 2, 'wall': 3}
actions = {'forward': 0, 'backward': 1, 'rotate_left': 2, 'rotate_right': 3, 'shoot': 4}
debug = True


def normalize_coordinate(value):
    return int(np.floor(value / 40))
