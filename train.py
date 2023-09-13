import matplotlib.pyplot as plt
import numpy as np

from loss_functions import MSE_Loss
from model import Model
from optimizers import Gradient_Decent


def main():
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    ground_truth = np.array([[0], [1], [1], [0]])

    loss_function = MSE_Loss()
    model = Model(input_data[0].shape)
    learning_rate = 0.01
    optimizer = Gradient_Decent(learning_rate, model)

    losses = []

    # Train
    epochs = 200
    for epoch in range(epochs):
        for input, gt in zip(input_data, ground_truth):
            prediction = model(input)
            loss = loss_function(prediction, gt)
            losses.append(loss)
            print(model.model[0].weights, prediction)
            model.zero_grad()
            model.backward(loss_function)
            optimizer.step()
    print()
    print(prediction, ground_truth)

    plt.plot(losses)
    plt.show()


main()
