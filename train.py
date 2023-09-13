import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from loss_functions import MSE_Loss
from model import Model
from optimizers import Gradient_Decent
from draw_graph import Nngraph


DEBUG = True


def main():
    input_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    ground_truth = np.array([[0.0], [1.0], [1.0], [2.0]])
    #input_data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    #ground_truth = np.array([[0], [0.2], [0.4], [0.6]])

    loss_function = MSE_Loss()
    model = Model(len(input_data[0]), len(ground_truth[0]), loss_function)
    learning_rate = 0.001
    optimizer = Gradient_Decent(learning_rate, model)

    # Train
    epochs = 50
    losses = []
    for epoch in tqdm(range(epochs)):
        for input, gt in zip(input_data, ground_truth):
            if DEBUG:
                nngraph = Nngraph(model, loss_function, f"graph_{epoch:04}")
                nngraph.draw_graph()
            prediction = model(input)
            loss = loss_function(prediction, gt)
            losses.append(loss)
            model.zero_grad()
            model.backward()
            optimizer.step()
    print()
    print("Predictions")
    print("GT, prediction")
    for input, gt in zip(input_data, ground_truth):
        print(input, gt, model(input))

    plt.plot(losses)
    plt.show()


main()
