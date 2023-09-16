from value import Value


class GradientDecent():
    def __init__(self, learning_rate, model):
        self.learning_rate = learning_rate
        self.model = model

    def step(self):
        for layer in self.model.model:
            for neuron in layer.neurons:
                # Dense layer
                if hasattr(neuron, "weights"):
                    neuron.weights = Value(neuron.weights - self.learning_rate * neuron.weights.grad)
                    if neuron.bias:
                        neuron.bias = Value(neuron.bias - self.learning_rate * neuron.bias.grad)

