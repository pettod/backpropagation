class Base_Model():
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def backward(self):
        self.loss_function.backward()
        output_neurons = [self.loss_function]
        for layer in reversed(self.model):
            for input_neuron in layer.neurons:
                for output_neuron in output_neurons:
                    input_neuron.backward(output_neuron)
            output_neurons = layer.neurons

    def zero_grad(self):
        self.loss_function.zero_grad()
        for layer in self.model:
            for neuron in layer.neurons:
                neuron.zero_grad()

    def forward(self, *args, **kwds):
        pass
