import numpy as np
from neuron import Neuron


class Layer:
    def __init__(self, inputs, outputs):
        self.neurons = [Neuron(inputs) for i in range(outputs)]

    def __call__(self, input):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(input))
        return np.array(outputs)
