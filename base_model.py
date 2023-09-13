class Base_Model():
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def backward(self, loss_function):
        loss_function.backward()
        output_neuron = loss_function
        for neuron in self.model:
            neuron.backward(output_neuron)

    def zero_grad(self):
        for neuron in self.model:
            neuron.zero_grad()

    def forward(self, *args, **kwds):
        pass
