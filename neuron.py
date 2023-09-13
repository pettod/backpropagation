import numpy as np
from value import Value
from weight_initialization import initialize_weights


class Neuron():
    def __init__(self, number_of_inputs, name, bias=False):
        self.weights = Value(initialize_weights((number_of_inputs)))
        self.bias = Value(initialize_weights((1))) if bias else None
        self.input = np.zeros((number_of_inputs))
        self.output = 0.0
        self.grad = 0.0
        self.name = name

    def __call__(self, input):
        self.input = input
        self.output = np.sum(self.weights * input)
        if self.bias:
            self.output += self.bias
        return self.output

    def zero_grad(self):
        self.grad = 0.0
        self.weights.zero_grad()
        if self.bias:
            self.bias.zero_grad()

    def backward(self, output_neuron):
        # (d loss / d neuron) = (d loss / d output) * (d output / d neuron)
        self.weights.backward(self.input, output_neuron)
        self.grad += np.sum(self.weights.grad)
        if self.bias:
            self.bias.backward(1.0, output_neuron)
            self.grad += self.bias.grad
