import numpy as np


class MSE_Loss():
    def __init__(self):
        self.loss = 0.0
        self.grad = 1.0
        self.input_grad = 0.0
        self.input = 0.0
        self.name = "loss_MSE"
        self.y_true = ""

    def __call__(self, y_pred, y_true):
        self.input = y_pred
        self.y_true = y_true

        # Important to have pred - true
        # Otherwise gradient should be multiplied with -1
        self.loss = float(np.sum((y_pred - y_true)**2))
        return self.loss

    def mse_derivative(self, x):
        return 2 * x

    def backward(self):
        self.input_grad += float(np.sum(self.mse_derivative((self.input - self.y_true))))

    def zero_grad(self):
        self.input_grad = 0.0
