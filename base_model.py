class Base_Model():
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def backward(self):
        self.loss_function.backward()
        output_neurons = []
        for i, layer in enumerate(reversed(self.model)):
            for j, input_neuron in enumerate(layer.neurons):
                if len(output_neurons) == 0:
                    input_neuron.grad += self.loss_function.input_grad
                for output_neuron in output_neurons:
                    input_neuron.grad += output_neuron.weights.data[j] * output_neuron.weights.grad[j]
                input_neuron.weights.grad += input_neuron.input * input_neuron.grad
                if input_neuron.bias:
                    input_neuron.bias.grad += input_neuron.grad
            output_neurons = layer.neurons

    def zero_grad(self):
        self.loss_function.zero_grad()
        for layer in self.model:
            for neuron in layer.neurons:
                neuron.zero_grad()

    def forward(self, *args, **kwds):
        pass
