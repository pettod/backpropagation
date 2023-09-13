import graphviz
from model import Model
from loss_functions import MSE_Loss


def main():
    model = Model(3, 1)
    loss_function = MSE_Loss()

    dot = graphviz.Digraph(comment="ML model graph", filename="graph.gv")
    layers = model.model
    for i, layer in enumerate(layers):
        for j, neuron in enumerate(layer.neurons):
            dot.node(neuron.name, neuron.name)
            if i == len(layers) - 1:
                dot.edge(neuron.name, loss_function.name)
            else:
                for next_layer_neuron in layers[i+1].neurons:
                    dot.edge(neuron.name, next_layer_neuron.name)
    dot.view()


main()
