import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from loss_functions import CrossEntropyLoss
from optimizers import GradientDecent
from draw_graph import Nngraph
from value import Value
from activations import ReLU, Softmax
from base_model import BaseModel
from dense import Dense


DEBUG = False
DATA_PATH = "mnist_train.csv"


class Model(BaseModel):
    def __init__(
            self,
            number_of_inputs,
            number_of_outputs,
            number_layers,
            features,
            bias,
            loss_function,
        ):
        if number_layers == 1:
            features = number_of_outputs
        self.loss_function = loss_function
        self.model = []
        for i in range(number_layers):
            if i == 0:
                layer = Dense(number_of_inputs, features, f"layer_{i}", bias)
                activation = ReLU(features)
            elif i < number_layers - 2:
                layer = Dense(features, features, f"layer_{i}", bias)
                activation = ReLU(features)
            else:
                layer = Dense(features, number_of_outputs, f"layer_{i}", bias)
                activation = Softmax(number_of_outputs)
            self.model.append(layer)
            self.model.append(activation)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


def targetFunction(input_data):
    def f(x):
        return 10 * [0.0]
    return np.array([f(input_sample) for input_sample in input_data])


def main():
    # Data
    number_of_samples = 100
    input_data = []
    ground_truth = []
    with open(DATA_PATH, newline="\n") as csvfile:
        mnist = csv.reader(csvfile, delimiter=",")
        for i, row in tqdm(enumerate(mnist)):
            if i == 0:
                continue
            ground_truth.append(int(row[0]))
            input_sample = []
            for pixel in row[1:]:
                normalized_input = float(pixel) / 255 - 0.5
                input_sample.append(Value(np.array(normalized_input)))
            input_data.append(input_sample)
            if i >= number_of_samples:
                break

    # One hot encode GT
    ground_truth = np.eye(10)[ground_truth]

    # Model
    number_of_inputs = len(input_data[0])
    number_of_outputs = len(ground_truth[0])
    number_layers = 4
    features = 100
    bias = True
    loss_function = CrossEntropyLoss()
    model = Model(
        number_of_inputs,
        number_of_outputs,
        number_layers,
        features,
        bias,
        loss_function,
    )
    learning_rate = 0.01
    optimizer = GradientDecent(learning_rate, model)

    # Train
    nngraph = Nngraph(model, loss_function)
    epochs = 10
    losses = []
    for epoch in range(1, 1+epochs):
        progress_bar = trange(len(input_data), leave=False)
        progress_bar.set_description(f" Epoch {epoch}/{epochs}")
        for i, (input, gt) in zip(progress_bar, zip(input_data, ground_truth)):
            prediction = model(input)
            loss = loss_function(prediction, gt)
            losses.append(loss)
            model.zeroGrad()
            model.backward()
            if DEBUG:
                nngraph.drawGraph(filename=f"graph_{epoch:04}_{i:04}")
            optimizer.step()
            progress_bar.display("loss: {:.4}".format(loss), 1)

    if True:
        print()
        print("Predictions")
        print("GT, prediction")
        for input, gt in zip(input_data, ground_truth):
            preds = [round(pred.data, 2) for pred in model(input)]
            print(gt, preds)

    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    if False:
        nngraph.drawGraph(view=True, filename=f"graph_final")


main()
