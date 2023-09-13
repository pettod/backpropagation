import numpy as np
from value import Value
from weight_initialization import initialize_weights


class Neuron():
    def __init__(self, input_shape, bias=False):
        self.weights = Value(initialize_weights(input_shape))
        self.bias = Value(initialize_weights((1))) if bias else None
        self.input = np.zeros((input_shape))

    def __call__(self, input):
        self.input = input
        output = np.sum(input * self.weights)
        if self.bias:
            output += self.bias
        return output

    def zero_grad(self):
        self.weights.zero_grad()
        if self.bias:
            self.bias.zero_grad()

    def backward(self, output_neuron):
        # (d loss / d neuron) = (d loss / d output) * (d output / d neuron)
        self.weights.backward(self.input, output_neuron)
        if self.bias:
            self.bias.backward(1.0, output_neuron)
