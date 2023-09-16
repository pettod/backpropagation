from activations import ReLU
from base_model import Base_Model
from layer import Layer


class Model(Base_Model):
    def __init__(
            self,
            number_of_inputs,
            number_of_outputs,
            number_layers,
            features,
            bias,
            loss_function,
        ):
        if number_layers == 1:
            features = number_of_outputs
        self.loss_function = loss_function
        self.model = []
        for i in range(number_layers):
            if i == 0:
                layer = Layer(number_of_inputs, features, f"layer_{i}", bias)
                activation = ReLU(features)
            elif i == number_layers - 1:
                layer = Layer(features, number_of_outputs, f"layer_{i}", bias)
                activation = ReLU(number_of_outputs)
            else:
                layer = Layer(features, features, f"layer_{i}", bias)
                activation = ReLU(features)
            self.model.append(layer)
            self.model.append(activation)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
