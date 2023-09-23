import math


class BaseActivation():
    def inputMax(self, input):
        input_max = None
        for value in input:
            if input_max is None:
                input_max = value.data
            elif value.data > input_max:
                input_max = value.data
        return input_max

    def inputSum(self, input):
        input_sum = 0.0
        for value in input:
            input_sum += value.data
        return input_sum

    def backward(self):
        for neuron in self.neurons:
            neuron.input.grad += self.derivative(neuron)


class Activation():
    def __init__(self):
        self.input = ""
        self.data = 0.0
        self.grad = 0.0

    def zeroGrad(self):
        self.grad = 0.0


class ReLU(BaseActivation):
    def __init__(self, inputs):
        self.neurons = [Activation() for i in range(inputs)]

    def __call__(self, input):
        for neuron, value in zip(self.neurons, input):
            neuron.input = value
            neuron.data = max(0, value.data)
        return self.neurons

    def derivative(self, neuron):
        return neuron.grad if neuron.data > 0.0 else 0.0


class Softmax(BaseActivation):
    def __init__(self, inputs):
        self.neurons = [Activation() for i in range(inputs)]
        self.outputs = []

    def __call__(self, input):
        input_max = self.inputMax(input)
        input_sum = self.inputSum(input)

        # Subtracting the maximum value for numerical stability
        for neuron, value in zip(self.neurons, input):
            neuron.input = value
            softmax_value = math.exp(value.data - input_max) / input_sum
            neuron.data = softmax_value
            self.outputs.append(softmax_value)
        return self.neurons

    def derivative(self, neuron):
        return neuron.data * (1.0 - neuron.data)
