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
        self.output = float(self.output)
        return self.output

    def backward(self):
        # Compute grads only for hidden layers
        if type(self.input) == Value:

            # Propagate input neurons grads
            for i in range(self.input):
                self.input[i].grad += float(self.weights.data[i] * self.weights.grad[i])

        # Set neuron weights grads
        self.weights.grad += self.input * self.grad

        # Set neuron bias grad
        if self.bias:
            self.bias.grad += float(self.grad)

    def zero_grad(self):
        self.grad = 0.0
        self.weights.zero_grad()
        if self.bias:
            self.bias.zero_grad()
