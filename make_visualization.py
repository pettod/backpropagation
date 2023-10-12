import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from glob import glob
import cv2
from PIL import Image, ImageDraw, ImageFont

from loss_functions import MSE
from model import Model
from optimizers import GradientDecent
from draw_graph import Nngraph
from value import Value


DEBUG = True


def targetFunction(input_data):
    def f(x):
        return float(max(0, (0.3*x[0] + 0.7*x[1]) + 0.2))
    return np.array([[f(input_sample)] for input_sample in input_data])


def main():
    seed = 42348
    random.seed(seed)
    np.random.seed(seed)

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
    epochs = 30
    losses = []
    avg_losses = []
    batch_losses = []
    for epoch in tqdm(range(epochs)):
        for i, (input, gt) in enumerate(zip(input_data, ground_truth)):
            prediction = model(input)
            loss = loss_function(prediction, gt)
            losses.append(loss)
            batch_losses.append(loss)
            model.zeroGrad()
            model.backward()
            if DEBUG and i % 8 == 0:
                avg_losses.append(sum(batch_losses) / len(batch_losses))
                batch_losses = []
                plt.close()
                plt.plot(avg_losses)
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.ylim(-0.01)
                plt.title("Loss")
                plt.savefig("plot.png")
                nngraph.drawGraph(filename=f"graph_{epoch:04}_{i:04}")
                plot_image = cv2.imread("plot.png")
                graph_image = cv2.imread(f"graphs/graph_{epoch:04}_{i:04}.gv.png")
                w = int(graph_image.shape[1] / 2.5)
                h = int((w / plot_image.shape[1]) * plot_image.shape[0])
                plot_image = cv2.resize(plot_image, (w, h))
                white_block = 255 + np.zeros((h, graph_image.shape[1] - w, 3), dtype=np.uint8)
                
                #white_block = Image.fromarray(white_block)
                #draw = ImageDraw.Draw(white_block)
                #font = ImageFont.truetype("Gidole-Regular.ttf", size=10)
                #font = ImageFont.load_default()
                #color = (0,0,0,0)
                #draw.text((50, 100), "test", font=font, fill=color)
                #white_block = np.array(white_block)

                image = np.concatenate([
                    graph_image,
                    np.concatenate([white_block, plot_image], axis=1)
                ], axis=0)

                cv2.circle(image, (890, 1359), 60, (156,156,230), -1)
                cv2.circle(image, (1118, 1359), 60, (156,156,230), -1)
                cv2.circle(image, (1347, 1359), 60, (156,156,230), -1)
                cv2.circle(image, (890, 1459), 60, (255,217,179), -1)
                cv2.circle(image, (1118, 1459), 60, (255,217,179), -1)
                cv2.circle(image, (1347, 1459), 60, (255,217,179), -1)

                weights = model.model[0].neurons[0].weights.data
                bias = model.model[0].neurons[0].bias.data

                # Pred formula
                cv2.putText(
                    image,
                    "max(0, {:.2f}i  + {:.2f}i  + {:.2f})".format(weights[0], weights[1], bias[0]),
                    (650, 1375), cv2.FONT_HERSHEY_SIMPLEX,  
                    1.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    "0",
                    (957, 1390), cv2.FONT_HERSHEY_SIMPLEX,  
                    1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    "1",
                    (1185, 1390), cv2.FONT_HERSHEY_SIMPLEX,  
                    1, (0, 0, 0), 2, cv2.LINE_AA)

                # Pred text
                cv2.putText(
                    image,
                    "Prediction:",
                    (230, 1375), cv2.FONT_HERSHEY_SIMPLEX,  
                    1.5, (0, 0, 0), 4, cv2.LINE_AA)

                # GT formula
                cv2.putText(
                    image,
                    "max(0, 0.30i  + 0.70i  + 0.20)",
                    (650, 1475), cv2.FONT_HERSHEY_SIMPLEX,  
                    1.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    "0",
                    (957, 1490), cv2.FONT_HERSHEY_SIMPLEX,  
                    1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    "1",
                    (1185, 1490), cv2.FONT_HERSHEY_SIMPLEX,  
                    1, (0, 0, 0), 2, cv2.LINE_AA)

                # GT text
                cv2.putText(
                    image,
                    "Target function:",
                    (230, 1475), cv2.FONT_HERSHEY_SIMPLEX,  
                    1.5, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.imwrite(f"graphs/graph_{epoch:04}_{i:04}.gv.png", image)

            optimizer.step()
    print()
    print("Predictions")
    print("Input, GT, Prediction")
    for input, gt in zip(input_data, ground_truth):
        preds = [pred.data for pred in model(input)]
        print(input, gt, preds)

    new_losses = []
    batch_size = 8
    for i in range(0, len(losses)-batch_size+1, batch_size):
        new_losses.append(sum(losses[i:i+batch_size]) / batch_size)
    print("len(new_losses) =", len(new_losses))
    plt.plot(new_losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    #plt.show()

    last_image_name = sorted(glob("graphs/*.png"))[-1]
    graph_image = cv2.imread(last_image_name)
    for c in range(20):
        cv2.imwrite("{}_0_{}.png".format(last_image_name[:-4], c), graph_image)
    copies = 6
    thickness_max = 6
    for thickness in range(1, thickness_max):
        cv2.circle(graph_image, (665, 461), 50, (44,44,191), thickness)  # Weight 0
        cv2.circle(graph_image, (890, 1359), 60, (44,44,191), thickness)  # Weight 0
        cv2.circle(graph_image, (890, 1459), 60, (199,131,54), thickness)  # Weight 0
        cv2.imwrite("{}_1_{}.png".format(last_image_name[:-4], thickness), graph_image)
    for c in range(copies):
        cv2.imwrite("{}_1_{}_{}.png".format(last_image_name[:-4], thickness, c), graph_image)
    for thickness in range(1, thickness_max):
        cv2.circle(graph_image, (665, 689), 50, (44,44,191), thickness)  # Weight 1
        cv2.circle(graph_image, (1118, 1359), 60, (44,44,191), thickness)  # Weight 1
        cv2.circle(graph_image, (1118, 1459), 60, (199,131,54), thickness)  # Weight 1
        cv2.imwrite("{}_2_{}.png".format(last_image_name[:-4], thickness), graph_image)
    for c in range(copies):
        cv2.imwrite("{}_2_{}_{}.png".format(last_image_name[:-4], thickness, c), graph_image)
    for thickness in range(1, thickness_max):
        cv2.circle(graph_image, (593, 233), 50, (44,44,191), thickness)  # Bias
        cv2.circle(graph_image, (1347, 1359), 60, (44,44,191), thickness)  # Bias
        cv2.circle(graph_image, (1347, 1459), 60, (199,131,54), thickness)  # Bias
        cv2.imwrite("{}_3_{}.png".format(last_image_name[:-4], thickness), graph_image)

    for c in range(40):
        cv2.imwrite("{}_4_{:02d}.png".format(last_image_name[:-4], c), graph_image)


main()
