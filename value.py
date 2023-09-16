import numpy as np


class Value():
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros(self.data.shape)

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

    def __getitem__(self, item):
         return self.data[item]

    def __iter__(self):
        for element in self.data:
            yield element

    def __len__(self):
        return len(self.data)

    def zeroGrad(self):
        self.grad = np.zeros(self.data.shape)

    def backward(self, input_data, output_neuron):
        self.grad += input_data * output_neuron.grad
