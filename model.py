from base_model import Base_Model
from layer import Layer


class Model(Base_Model):
    def __init__(self, number_of_inputs, number_of_outputs):
        self.model = [
            Layer(number_of_inputs, 2, "layer_0"),
            Layer(2, number_of_outputs, "layer_1"),
        ]

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
