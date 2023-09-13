from base_model import Base_Model
from neuron import Neuron


class Model(Base_Model):
    def __init__(self, number_of_inputs):
        self.model = [
            Neuron(number_of_inputs),
        ]

    def forward(self, x):
        for neuron in self.model:
            x = neuron(x)
        return x
