import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from loss_functions import MSE_Loss
from model import Model
from optimizers import Gradient_Decent
from draw_graph import Nngraph


DEBUG = False


def main():
    #input_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    #ground_truth = np.array([[0.0], [0.5], [-0.5], [0.0]])
    #input_data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    #ground_truth = np.array([[0.0], [0.2], [0.4], [0.6]])
    input_data = np.array([[0.4, 0.3]])
    ground_truth = np.array([[0.35]])

    # Normalize input data
    #input_data /= input_data.max()
    #input_data -= input_data.mean()

    # Normalize ground truth
    #ground_truth /= ground_truth.max()
    #ground_truth -= ground_truth.mean()
    
    number_of_inputs = len(input_data[0])
    number_of_outputs = len(ground_truth[0])
    number_layers = 5
    features = 2
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
    learning_rate = 0.01
    optimizer = Gradient_Decent(learning_rate, model)

    # Train
    epochs = 10000
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
    print()
    print("Predictions")
    print("GT, prediction")
    for input, gt in zip(input_data, ground_truth):
        print(input, gt, model(input))

    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


main()
