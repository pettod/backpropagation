import graphviz
from activations import Activation
from loss_functions import MSE
from model import Model
from neuron import Neuron


FIXED_SIZE = "true"
NODE_ATTR = {
    "activation": {
        "shape": "record",
        "style": "rounded,filled",
        "fixedsize": FIXED_SIZE,
        "width": "1.50",
        "fillcolor": "#DAE8FC",
        "color": "#6C8EBF",
        "fontsize": "10pt",
    },
    "bias": {
        "shape": "record",
        "style": "rounded,filled",
        "fixedsize": FIXED_SIZE,
        "width": "0.80",
        "fillcolor": "#E1D5E7",
        "color": "#9673A6",
        "fontsize": "10pt",
    },
    "ground_truth": {
        "shape": "record",
        "style": "filled",
        "fixedsize": FIXED_SIZE,
        "width": "1.00",
        "fillcolor": "#FFD2B0",
        "color": "#C48E00",
        "fontsize": "10pt",
    },
    "input": {
        "shape": "record",
        "style": "filled",
        "fixedsize": FIXED_SIZE,
        "width": "0.70",
        "fillcolor": "#FFF2CC",
        "color": "#D6B656",
        "fontsize": "10pt",
    },
    "layer": {
        "style": "rounded,filled",
        "fillcolor": "#EBEBEB",
        "color": "#666666",
    },
    "sum": {
        "shape": "record",
        "style": "rounded,filled",
        "fixedsize": FIXED_SIZE,
        "width": "1.20",
        "fillcolor": "#F8CECC",
        "color": "#B85450",
        "fontsize": "10pt",
    },
    "weight": {
        "shape": "record",
        "style": "rounded,filled",
        "fixedsize": FIXED_SIZE,
        "width": "1.20",
        "fillcolor": "#D5E8D4",
        "color": "#82B366",
        "fontsize": "10pt",
    },
}


class Nngraph():
    def __init__(self, model, loss_function, filename="graph", weight_boxes=True):
        self.dot = self.initDot(filename)
        self.layers = model.model
        self.loss_function = loss_function
        self.weight_boxes = weight_boxes

    def initDot(self, filename):
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
        dot.edge(start_node, end_node, headport="w", tailport="e")

    def node(self, node_name, label, attr):
        self.dot.node(node_name, label, **attr)

    def inputNodes(self, neuron, i):
        for k, input in enumerate(neuron.input):
            previous_layer_node_name = f"layer_{i}_out_{k}"
            input_label = r"input\n{:.2f}".format(float(input.data))
            self.node(previous_layer_node_name, input_label, NODE_ATTR["input"])

    def singleLayerNode(self, current_node_name, neuron):
        # Node sum
        sum_label = "{%s}" % r"+ | data {:.2f}\ngrad {:.2f}".format(float(neuron.data), float(neuron.grad))
        self.node(current_node_name, sum_label, NODE_ATTR["sum"])

        # Node bias
        if neuron.bias:
            bias_label = r"bias {:.2f}\ngrad {:.2f}".format(float(neuron.bias.data), float(neuron.bias.grad))
            self.node(f"{current_node_name}_bias", bias_label, NODE_ATTR["bias"])

    def edgesFromPreviousLayerToCurrent(self, current_node_name, neuron, i, layer_index):
        neuron_bias_edge_added = False
        for k, weight in enumerate(neuron.weights.data):
            previous_layer_node_name = f"layer_{i}_out_{k}"

            # Box nodes for weights
            if self.weight_boxes:

                # Weight node
                weight_node_name = f"{previous_layer_node_name}_{current_node_name}"
                weight_label = "{%s}" % r"* | weight {:.2f}\ngrad {:.2f}".format(weight, neuron.weights.grad[k])
                self.node(weight_node_name, weight_label, NODE_ATTR["weight"])

                # Connect weight to the input
                self.edge(self.dot, previous_layer_node_name, weight_node_name)

                # Add layer cluster
                with self.dot.subgraph(name=f"cluster_{i}") as cluster:
                    cluster.attr(label=f"Layer {layer_index}", **NODE_ATTR["layer"])

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
                    label="w {:.2f}\ng {:.2f}".format(weight, neuron.weights.grad[k]),
                    fontsize="10pt",
                )

    def activationFunction(self, current_node_name, neuron, activation_name):
        activation_label = "{%s}" % r"{} | data {:.2f}\ngrad {:.2f}".format(activation_name, float(neuron.data), float(neuron.grad))
        self.node(current_node_name, activation_label, NODE_ATTR["activation"])

    def edgeFromActivationToNeuron(self, current_node_name, i, j):
        # Add layer cluster
        with self.dot.subgraph(name=f"cluster_{i-1}") as cluster:
            previous_layer_node_name = f"layer_{i}_out_{j}"
            self.edge(cluster, previous_layer_node_name, current_node_name)

    def loss(self):
        loss_label = "{%s}" % r"{} | loss {:.2f}\ngrad {:.2f}".format(type(self.loss_function).__name__, self.loss_function.loss, self.loss_function.grad)
        self.node("loss", loss_label, NODE_ATTR["activation"])

    def groundTruth(self):
        ground_truth_label = r"ground truth"
        for gt_element in self.loss_function.y_true:
            ground_truth_label += r"\n{:.2f}".format(gt_element)
        self.node("ground_truth", ground_truth_label, NODE_ATTR["ground_truth"])
        self.edge(self.dot, "ground_truth", "loss")

    def drawGraph(self, view=False, filename=None):
        if filename:
            self.dot = self.initDot(filename)
        layer_index = 1
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                current_node_name = f"layer_{i+1}_out_{j}"
                if i == 0 and j == 0:
                    self.inputNodes(neuron, 0)
                if type(neuron) == Neuron:
                    self.singleLayerNode(current_node_name, neuron)
                    self.edgesFromPreviousLayerToCurrent(current_node_name, neuron, i, layer_index)
                elif type(neuron) == Activation:
                    self.activationFunction(current_node_name, neuron, type(layer).__name__)
                    self.edgeFromActivationToNeuron(current_node_name, i, j)
                if i == len(self.layers) - 1:
                    self.edge(self.dot, current_node_name, "loss")
            if type(neuron) != Activation:
                layer_index += 1
        self.loss()
        self.groundTruth()
        self.dot.render(directory="graphs", view=view)


if __name__ == "__main__":
    number_of_inputs = 2
    number_of_outputs = 1
    number_layers = 2
    features = 3
    bias = True
    loss_function = MSE()
    model = Model(
        number_of_inputs,
        number_of_outputs,
        number_layers,
        features,
        bias,
        loss_function,
    )
    nngraph = Nngraph(model, loss_function)
    nngraph.drawGraph(view=True)
