import matplotlib.pyplot as plt
from tqdm import trange

from draw_graph import Nngraph


def trainModel(
        model, input_data, ground_truth, optimizer, epochs=10, print_predictions=True,
        draw_graph_after_iterations=False, draw_graph_after_training=True,
    ):
    loss_function = model.loss_function
    nngraph = Nngraph(model, loss_function)
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
            if draw_graph_after_iterations:
                nngraph.drawGraph(filename=f"graph_{epoch:04}_{i:04}")
            optimizer.step()
            progress_bar.display("loss: {:.4}".format(loss), 1)

    if print_predictions:
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

    if draw_graph_after_training:
        nngraph.drawGraph(view=True, filename=f"graph_final")
