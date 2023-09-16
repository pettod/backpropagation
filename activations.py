class Activation():
    def __init__(self):
        self.input = ""
        self.data = 0.0
        self.grad = 0.0

    def zeroGrad(self):
        self.grad = 0.0


class ReLU():
    def __init__(self, inputs):
        self.neurons = [Activation() for i in range(inputs)]

    def __call__(self, input):
        for neuron, value in zip(self.neurons, input):
            neuron.input = value
            neuron.data = max(0, value.data)
        return self.neurons

    def backward(self):
        for neuron in self.neurons:
            if neuron.data > 0:
                neuron.input.grad += neuron.grad
