batch_size = 10
logging_threshold = 10

max_frame_size = 300
frame_height = 227
frame_width = 227
frame_channels = 3
context_channels = 5



from enum import Enum

class Subset(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Temporal_Direction(Enum):
    BEFORE = 0
    AFTER = 1




train_iters = 2000000
learning_rate = 10e-4
num_filters = 23

