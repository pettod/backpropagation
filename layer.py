import numpy as np
from neuron import Neuron


class Layer:
    def __init__(self, inputs, outputs, name, bias=False):
        self.name = name
        self.inputs = inputs
        self.datas = outputs
        self.neurons = [Neuron(inputs, f"{self.name}_neuron_{i}", bias) for i in range(outputs)]

    def __call__(self, input):
        #if type(input) != list:
        #    input 
        for neuron in self.neurons:
            neuron(input)
        return self.neurons

    def backward(self):
        for neuron in self.neurons:
            neuron.backward()
