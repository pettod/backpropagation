import numpy as np


class Neuron():
    def __init__(self, input_shape, bias=False):
        self.weights = np.random.uniform(-1, 1, (input_shape))
        self.bias = np.random.uniform(-1, 1) if bias else None
        self.grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros((1))
        self.input = np.zeros((input_shape))

    def __call__(self, input):
        self.input = input
        output = np.sum(input * self.weights)
        if self.bias:
            output += self.bias
        return output

    def zero_grad(self):
        self.grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros((1))

    def backward(self, output_neuron):
        # (d loss / d neuron) = (d loss / d output) * (d output / d neuron)
        self.grad += self.input * output_neuron.grad
