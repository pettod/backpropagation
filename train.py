import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from loss_functions import MSE_Loss
from model import Model
from optimizers import Gradient_Decent
from draw_graph import Nngraph


DEBUG = False


def target_function(input_data):
    def f(x):
        return (0.3*x[0] + 0.7*x[1]) + 0.2
    return np.array([[f(input_sample)] for input_sample in input_data])


def main():
    input_data = np.random.uniform(-1, 1, (50, 2))
    ground_truth = target_function(input_data)

    number_of_inputs = len(input_data[0])
    number_of_outputs = len(ground_truth[0])
    number_layers = 1
    features = 2
    bias = True
    loss_function = MSE_Loss()
    model = Model(
        number_of_inputs,
        number_of_outputs,
        number_layers,
        features,
        bias,
        loss_function,
    )
    learning_rate = 0.01
    optimizer = Gradient_Decent(learning_rate, model)

    # Train
    epochs = 1000
    losses = []
    for epoch in tqdm(range(epochs)):
        for i, (input, gt) in enumerate(zip(input_data, ground_truth)):
            prediction = model(input)
            loss = loss_function(prediction, gt)
            losses.append(loss)
            model.zero_grad()
            model.backward()
            if DEBUG:
                nngraph = Nngraph(model, loss_function, f"graph_{epoch:04}_{i:04}")
                nngraph.draw_graph()
            optimizer.step()
    #print()
    #print("Predictions")
    #print("GT, prediction")
    #for input, gt in zip(input_data, ground_truth):
    #    print(input, gt, model(input))

    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    nngraph = Nngraph(model, loss_function, f"graph_final")
    nngraph.draw_graph(view=True)

main()
