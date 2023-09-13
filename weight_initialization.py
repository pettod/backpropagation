import numpy as np


def initialize_weights(input_shape):
    return np.random.uniform(-1, 1, (input_shape))
