import graphviz
from model import Model
from loss_functions import MSE_Loss
from neuron import Neuron
from activations import Activation


class Nngraph():
    def __init__(self, model, loss_function, filename="graph", weight_boxes=True):
        self.dot = self.init_dot(filename)
        self.layers = model.model
        self.loss_function = loss_function
        self.weight_boxes = weight_boxes

    def init_dot(self, filename):
        return graphviz.Digraph(
            comment="ML model graph",
            filename=f"{filename}.gv",
            format="png",
            graph_attr={
                "rankdir": "LR",  # Left to right graph
                "dpi": "300",  # DPI for the png file
                "splines": "false",  # Straight lines/edges between the nodes
            },
        )

    def edge(self, dot, start_node, end_node):
        dot.edge(
            start_node,
            end_node,
            headport="w",
            tailport="e",
        )

    def add_input_nodes(self, neuron, i):
        for k, input in enumerate(neuron.input):
            previous_layer_node_name = f"layer_{i}_out_{k}"
            self.dot.node(
                previous_layer_node_name,
                r"input\n{:.2}".format(float(input.data)),
                shape="record",
                style="filled",
                fillcolor="#FFF2CC",
                color="#D6B656",
                fontsize="10pt",
            )

    def add_single_layer_node(self, current_node_name, j, neuron):
        node_text = r"+ | value {:.2}\ngrad {:.2}".format(float(neuron.data), float(neuron.grad))

        # Node
        self.dot.node(
            current_node_name,
            "{%s}" % node_text,
            shape="record",
            style="rounded,filled",
            fillcolor="#F8CECC",
            color="#B85450",
            fontsize="10pt",
        )

        # Node bias
        if neuron.bias:
            self.dot.node(
                "{}_bias".format(current_node_name),
                r"bias {:.2}\ngrad {:.2}".format(float(neuron.bias.data), float(neuron.bias.grad)),
                shape="record",
                style="rounded,filled",
                fillcolor="#E1D5E7",
                color="#9673A6",
                fontsize="10pt",
            )

    def add_edges_from_previous_layer_to_current(self, current_node_name, neuron, i, layer_index):
        neuron_bias_edge_added = False
        for k, weight in enumerate(neuron.weights.data):
            previous_layer_node_name = f"layer_{i}_out_{k}"

            # Create box nodes for weights
            if self.weight_boxes:

                # Create weight node
                weight_node_name = f"{previous_layer_node_name}_{current_node_name}"
                node_text = r"* | weight {:.2}\ngrad {:.2}".format(weight, neuron.weights.grad[k])
                self.dot.node(
                    weight_node_name,
                    "{%s}" % node_text,
                    shape="record",
                    style="rounded,filled",
                    fillcolor="#D5E8D4",
                    color="#82B366",
                    fontsize="10pt",
                )

                # Connect weight to the input
                self.edge(self.dot, previous_layer_node_name, weight_node_name)

                # Add layer cluster
                with self.dot.subgraph(name=f"cluster_{i}") as cluster:
                    cluster.attr(
                        label=f"Layer {layer_index}",
                        style="rounded,filled",
                        fillcolor="#EBEBEB",
                        color="#666666",
                    )

                    # Connect weight to the output node
                    self.edge(cluster, weight_node_name, current_node_name)

                    # Connect bias
                    if neuron.bias and not neuron_bias_edge_added:
                        neuron_bias_edge_added = True
                        self.edge(cluster, "{}_bias".format(current_node_name), current_node_name)

            # Add weight texts the edges
            else:
                self.dot.edge(
                    previous_layer_node_name,
                    current_node_name,
                    label="w {:.2}\ng {:.2}".format(weight, neuron.weights.grad[k]),
                    fontsize="10pt",
                )

    def add_activation_function(self, current_node_name, neuron, activation_name):
        node_text = r"{} | value {:.2}\ngrad {:.2}".format(activation_name, float(neuron.data), float(neuron.grad))
        self.dot.node(
            current_node_name,
            "{%s}" % node_text,
            shape="record",
            style="rounded,filled",
            fillcolor="#DAE8FC",
            color="#6C8EBF",
            fontsize="10pt",
        )

    def add_edge_from_activation_to_neuron(self, current_node_name, i, j):
        # Add layer cluster
        with self.dot.subgraph(name=f"cluster_{i-1}") as cluster:
            previous_layer_node_name = f"layer_{i}_out_{j}"
            self.edge(cluster, previous_layer_node_name, current_node_name)

    def add_loss(self, current_node_name):
        node_text = r"{} | loss {:.2}\ngrad {:.2}".format(
            type(self.loss_function).__name__, self.loss_function.loss, self.loss_function.grad)
        self.dot.node(
            "loss",
            "{%s}" % node_text,
            shape="record",
            style="rounded,filled",
            fillcolor="#DAE8FC",
            color="#6C8EBF",
            fontsize="10pt",
        )
        self.edge(self.dot, current_node_name, "loss")

    def add_ground_truth(self):
        ground_truth_label = r"ground truth"
        for gt_element in self.loss_function.y_true:
            ground_truth_label += r"\n{:.2}".format(gt_element)
        self.dot.node(
            "ground_truth",
            ground_truth_label,
            shape="record",
            style="filled",
            fillcolor="#FFD2B0",
            color="#C48E00",
            fontsize="10pt",
        )
        self.edge(self.dot, "ground_truth", "loss")

    def draw_graph(self, view=False, filename=None):
        if filename:
            self.dot = self.init_dot(filename)
        layer_index = 1
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                current_node_name = f"layer_{i+1}_out_{j}"
                if i == 0 and j == 0:
                    self.add_input_nodes(neuron, 0)
                if type(neuron) == Neuron:
                    self.add_single_layer_node(current_node_name, j, neuron)
                    self.add_edges_from_previous_layer_to_current(current_node_name, neuron, i, layer_index)
                elif type(neuron) == Activation:
                    self.add_activation_function(current_node_name, neuron, type(layer).__name__)
                    self.add_edge_from_activation_to_neuron(current_node_name, i, j)
            if type(neuron) != Activation:
                layer_index += 1
        self.add_loss(current_node_name)
        self.add_ground_truth()
        self.dot.render(directory="graphs", view=view)


if __name__ == "__main__":
    number_of_inputs = 2
    number_of_outputs = 1
    number_layers = 2
    features = 3
    bias = False
    loss_function = MSE_Loss()
    model = Model(
        number_of_inputs,
        number_of_outputs,
        number_layers,
        features,
        bias,
        loss_function,
    )
    nngraph = Nngraph(model, loss_function)
    nngraph.draw_graph(view=True)
