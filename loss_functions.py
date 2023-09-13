import numpy as np


class MSE_Loss():
    def __init__(self):
        self.loss = 0.0
        self.grad = 0.0

    def __call__(self, y_pred, y_true):
        self.loss = np.sum((y_true - y_pred)**2)
        return self.loss

    def backward(self):
        self.grad = 2 * self.loss
