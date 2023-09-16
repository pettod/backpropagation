import numpy as np
from value import Value
from weight_initialization import initialize_weights


class Neuron():
    def __init__(self, number_of_inputs, name, bias=False):
        self.weights = Value(initialize_weights((number_of_inputs)))
        self.bias = Value(initialize_weights((1))) if bias else None
        self.input = [Value(np.array(0.0)) for i in range(number_of_inputs)]
        self.data = 0.0
        self.grad = 0.0
        self.name = name

    def __call__(self, input):
        self.input = input
        self.data = 0.0
        for weight, input_value in zip(self.weights.data, self.input):
            self.data += weight * input_value.data
        if self.bias:
            self.data += self.bias
        self.data = float(self.data)
        return self.data

    def backward(self):
        # Propagate input neurons grads
        for i in range(len(self.input)):
            self.input[i].grad += float(self.weights.data[i] * self.weights.grad[i])

            # Set neuron weights grads
            self.weights.grad[i] += self.input[i].data * self.grad

        # Set neuron bias grad
        if self.bias:
            self.bias.grad += float(self.grad)

    def zero_grad(self):
        self.grad = 0.0
        self.weights.zero_grad()
        if self.bias:
            self.bias.zero_grad()
