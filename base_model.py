class BaseModel():
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def backward(self):
        for layer in reversed(self.model + [self.loss_function]):
            layer.backward()

    def zeroGrad(self):
        for layer in self.model:
            for neuron in layer.neurons:
                neuron.zeroGrad()

    def forward(self, *args, **kwds):
        pass
