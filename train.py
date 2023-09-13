import matplotlib.pyplot as plt
import numpy as np

from loss_functions import MSE_Loss
from model import Model
from optimizers import Gradient_Decent


def main():
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    ground_truth = np.array([[0], [0], [0], [0]])
    #input_data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    #ground_truth = np.array([[0], [0.2], [0.4], [0.6]])

    loss_function = MSE_Loss()
    model = Model(len(input_data[0]), len(ground_truth[0]))
    learning_rate = 0.001
    optimizer = Gradient_Decent(learning_rate, model)

    losses = []

    # Train
    epochs = 50
    for epoch in range(epochs):
        for input, gt in zip(input_data, ground_truth):
            prediction = model(input)
            loss = loss_function(prediction, gt)
            losses.append(loss)
            model.zero_grad()
            model.backward(loss_function)
            optimizer.step()
    print()
    print("Predictions")
    print("GT, prediction")
    for input, gt in zip(input_data, ground_truth):
        print(gt, model(input))

    plt.plot(losses)
    plt.show()


main()
