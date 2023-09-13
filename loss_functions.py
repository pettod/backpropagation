import numpy as np


class MSE_Loss():
    def __init__(self):
        self.loss = 0.0
        self.grad = 0.0
        self.input = 0.0
        self.name = "loss_MSE"

    def __call__(self, y_pred, y_true):
        self.input = y_pred
        self.loss = np.sum((y_true - y_pred)**2)
        return self.loss

    def mse_derivative(self, x):
        return 2 * x

    def backward(self):
        d_neuron = self.mse_derivative(self.loss) * self.loss
        self.grad = self.input * d_neuron
