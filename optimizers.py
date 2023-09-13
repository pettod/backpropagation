class Gradient_Decent():
    def __init__(self, learning_rate, model):
        self.learning_rate = learning_rate
        self.model = model

    def step(self):
        for neuron in self.model.model:
            neuron.weights -= self.learning_rate * neuron.grad
