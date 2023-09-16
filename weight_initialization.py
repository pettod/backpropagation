import numpy as np


def initializeWeights(input_shape):
    return np.random.uniform(-1, 1, (input_shape))
