import numpy as np


class Value():
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros(self.data.shape)
        self.backward = self.backward
        self.zero_grad = self.zero_grad

    def __call__(self):
        return self.data

    def __repr__(self):
        return "{}".format(self.data)

    def __add__(self, other):
        return self.data + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1.0 * other)

    def __rsub__(self, other):
        return self.__add__(-1.0 * other)

    def __mul__(self, other):
        return self.data * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        for i in range(other-1):
            self.data *= self.data
        return self.data

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)

    def backward(self, input_data, output_neuron):
        self.grad += input_data * output_neuron.grad


class Neuron():
    def __init__(self, input_shape, bias=False):
        self.weights = Value(np.random.uniform(-1, 1, (input_shape)))
        self.bias = Value(np.random.uniform(-1, 1, (1))) if bias else None
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
