import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from loss_functions import MSE
from model import Model
from optimizers import GradientDecent
from draw_graph import Nngraph
from value import Value


DEBUG = False


def targetFunction(input_data):
    def f(x):
        return float(max(0, (0.3*x[0] + 0.7*x[1]) + 0.2))
    return np.array([[f(input_sample)] for input_sample in input_data])


def main():
    # Data
    input_data = [
        [
            Value(np.random.uniform(-1, 1, (1))),
            Value(np.random.uniform(-1, 1, (1))),
        ] for i in range(50)]
    ground_truth = targetFunction(input_data)

    # Model
    number_of_inputs = len(input_data[0])
    number_of_outputs = len(ground_truth[0])
    number_layers = 1
    features = 2
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
    learning_rate = 0.01
    optimizer = GradientDecent(learning_rate, model)

    # Train
    nngraph = Nngraph(model, loss_function)
    epochs = 1000
    losses = []
    for epoch in tqdm(range(epochs)):
        for i, (input, gt) in enumerate(zip(input_data, ground_truth)):
            prediction = model(input)
            loss = loss_function(prediction, gt)
            losses.append(loss)
            model.zeroGrad()
            model.backward()
            if DEBUG:
                nngraph.drawGraph(filename=f"graph_{epoch:04}_{i:04}")
            optimizer.step()
    print()
    print("Predictions")
    print("GT, prediction")
    for input, gt in zip(input_data, ground_truth):
        preds = [pred.data for pred in model(input)]
        print(input, gt, preds)

    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    nngraph.drawGraph(view=True, filename=f"graph_final")

main()
