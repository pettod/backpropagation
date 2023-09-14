class Base_Model():
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def backward(self):
        for layer in reversed(self.model + [self.loss_function]):
            layer.backward()

    def zero_grad(self):
        self.loss_function.zero_grad()
        for layer in self.model:
            for neuron in layer.neurons:
                neuron.zero_grad()

    def forward(self, *args, **kwds):
        pass
