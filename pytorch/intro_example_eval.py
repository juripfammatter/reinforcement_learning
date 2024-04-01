import json
import sys
import os

import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
# import neural network architecture (needed for model)
from neural_networks.intro_example_NN import NeuralNetwork


def load_config() -> dict:
    """ Import JSON file and check existence"""
    config_file = sys.argv[1]
    if os.path.exists(config_file):
        print(f"Using {config_file} as config file")

    with open(config_file, "r") as file:
        config = json.load(file)

    print(f"Config file loaded: {config_file} with config:\n{json.dumps(config, indent=4)}")

    return config


def setup_model(config: dict) -> (NeuralNetwork, str):
    """ Model """
    # Get cpu, gpu or mps device for training.
    best_available_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {best_available_device} device")
    model_filename = config["model_filename"]
    model = torch.load(model_filename)
    print(model)

    return model, best_available_device


def load_dataset() -> any:
    """ Dataset """
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return test_data


if __name__ == "__main__":
    config = load_config()
    model, device = setup_model(config)
    test_data = load_dataset()

    """ Setup classes """
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    """ Evaluate performance """
    model.eval()
    rnd = (np.random.random(size=1) * len(test_data)).astype(int).item()
    x, y = test_data[rnd][0], test_data[rnd][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
