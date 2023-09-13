import graphviz
from model import Model
from loss_functions import MSE_Loss


def draw_graph(model, loss_function):
    dot = graphviz.Digraph(comment="ML model graph", filename="graph.gv")
    layers = model.model

    for i, layer in enumerate(layers):
        for j, neuron in enumerate(layer.neurons):

            # Add input nodes
            if i == 0 and j == 0:
                for k, input in enumerate(neuron.input):
                    previous_layer_node_name = f"layer_{i}_out_{k}"
                    dot.node(
                        previous_layer_node_name,
                        "input {:.2}\n".format(input),
                        style="filled",
                        color="#FDFFA5",
                    )

            # Add single layer node
            current_node_name = f"layer_{i+1}_out_{j}"
            dot.node(
                current_node_name,
                "value {:.2}\ngrad {:.2}".format(neuron.input[j], neuron.grad),
                style="filled",
                color="#FF4444",
            )

            # Add edges from previous layer to current
            for k, weight in enumerate(neuron.weights.data):
                previous_layer_node_name = f"layer_{i}_out_{k}"
                dot.edge(
                    previous_layer_node_name,
                    current_node_name,
                    label="w {:.2}\ng {:.2}".format(weight, neuron.weights.grad[k])
                )

    # Add loss
    dot.node(
        "loss",
        "loss {:.2}\ngrad {:.2}".format(loss_function.loss, loss_function.grad),
                        style="filled",
        color="#448BFF",
    )
    dot.edge(
        current_node_name,
        "loss",
        label="",
    )
    dot.view()


if __name__ == "__main__":
    loss_function = MSE_Loss()
    model = Model(2, 1, loss_function)
    draw_graph(model, loss_function)
