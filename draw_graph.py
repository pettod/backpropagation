import graphviz
from model import Model
from loss_functions import MSE_Loss


class Nngraph():
    def __init__(self, model, loss_function, filename="graph", weight_boxes=True):
        self.dot = graphviz.Digraph(
            comment="ML model graph",
            filename=f"{filename}.gv",
            format="png",
            graph_attr={"rankdir": "LR", "dpi": "300"},
        )
        self.layers = model.model
        self.loss_function = loss_function
        self.weight_boxes = weight_boxes

    def add_input_nodes(self, neuron, i):
        for k, input in enumerate(neuron.input):
            previous_layer_node_name = f"layer_{i}_out_{k}"
            self.dot.node(
                previous_layer_node_name,
                "input {:.2}\n".format(input),
                style="filled",
                fillcolor="#FFF2CC",
                color="#D6B656",
            )

    def add_single_layer_node(self, current_node_name, j, neuron):
        node_text = r"+ | value {:.2}\ngrad {:.2}".format(neuron.input[j], neuron.grad)
        self.dot.node(
            current_node_name,
            "{%s}" % node_text,
            shape="record",
            style="filled",
            fillcolor="#F8CECC",
            color="#B85450",
        )

    def add_edges_from_previous_layer_to_current(self, current_node_name, neuron, i):
        for k, weight in enumerate(neuron.weights.data):
            previous_layer_node_name = f"layer_{i}_out_{k}"

            # Create box nodes for weights
            if self.weight_boxes:
                weight_node_name = f"{previous_layer_node_name}_{current_node_name}"
                node_text = r"* | weight {:.2}\ngrad {:.2}".format(weight, neuron.weights.grad[k])
                self.dot.node(
                    weight_node_name,
                    "{%s}" % node_text,
                    shape="record",
                    style="filled",
                    fillcolor="#D5E8D4",
                    color="#82B366",
                    fontsize="10pt",
                )
                self.dot.edge(previous_layer_node_name, weight_node_name)

                # Add layer cluster
                with self.dot.subgraph(name=f"cluster_{i}") as cluster:
                    cluster.attr(
                        label=f"Layer {i+1}",
                        style="filled",
                        fillcolor="#EBEBEB",
                        color="#666666",
                    )
                    cluster.edge(weight_node_name, current_node_name)

            # Add weight texts the edges
            else:
                self.dot.edge(
                    previous_layer_node_name,
                    current_node_name,
                    label="w {:.2}\ng {:.2}".format(weight, neuron.weights.grad[k]),
                    fontsize="10pt",
                )

    def add_loss(self, current_node_name):
        self.dot.node(
            "loss",
            "loss {:.2}\ngrad {:.2}".format(
                self.loss_function.loss, self.loss_function.grad),
            style="filled",
            fillcolor="#DAE8FC",
            color="#6C8EBF",
        )
        self.dot.edge(current_node_name, "loss")

    def draw_graph(self):
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                current_node_name = f"layer_{i+1}_out_{j}"
                if i == 0 and j == 0:
                    self.add_input_nodes(neuron, i)
                self.add_single_layer_node(current_node_name, j, neuron)
                self.add_edges_from_previous_layer_to_current(current_node_name, neuron, i)
        self.add_loss(current_node_name)
        self.dot.render(directory="graphs", view=True)


if __name__ == "__main__":
    loss_function = MSE_Loss()
    model = Model(2, 1, loss_function)
    nngraph = Nngraph(model, loss_function)
    nngraph.draw_graph()
