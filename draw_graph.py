import graphviz
from model import Model
from loss_functions import MSE_Loss


def draw_graph(model, loss_function, weight_boxes=True):
    dot = graphviz.Digraph(
        comment="ML model graph",
        filename="graph.gv",
        graph_attr={"rankdir": "LR"},
    )
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
                        fillcolor="#FDFFA5",
                    )

            # Add single layer node
            current_node_name = f"layer_{i+1}_out_{j}"
            dot.node(
                current_node_name,
                "value {:.2}\ngrad {:.2}".format(neuron.input[j], neuron.grad),
                style="filled",
                fillcolor="#FF4444",
            )

            # Add edges from previous layer to current
            for k, weight in enumerate(neuron.weights.data):
                previous_layer_node_name = f"layer_{i}_out_{k}"

                # Create box nodes for weights
                if weight_boxes:
                    weight_node_name = f"{previous_layer_node_name}_{current_node_name}"
                    dot.node(
                        weight_node_name,
                        "weight {:.2}\ngrad {:.2}".format(weight, neuron.weights.grad[k]),
                        shape="box",
                        style="filled",
                        fillcolor="#64FF73",
                        fontsize="10pt",
                    )
                    dot.edge(previous_layer_node_name, weight_node_name)
                    dot.edge(weight_node_name, current_node_name)

                # Add weight texts the edges
                else:
                    dot.edge(
                        previous_layer_node_name,
                        current_node_name,
                        label="w {:.2}\ng {:.2}".format(weight, neuron.weights.grad[k]),
                        fontsize="10pt",
                    )

    # Add loss
    dot.node(
        "loss",
        "loss {:.2}\ngrad {:.2}".format(loss_function.loss, loss_function.grad),
        style="filled",
        fillcolor="#448BFF",
    )
    dot.edge(current_node_name, "loss")
    dot.view()


if __name__ == "__main__":
    loss_function = MSE_Loss()
    model = Model(2, 1, loss_function)
    draw_graph(model, loss_function)
