import json
import sys
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from neural_networks.intro_example_NN import NeuralNetwork


def load_config():
    """ Import JSON file and check existence"""
    config_file = sys.argv[1]
    if os.path.exists(config_file):
        print(f"Using {config_file} as config file")

    with open(config_file, "r") as file:
        config = json.load(file)

    print(f"Config file loaded: {config_file} with config:\n{json.dumps(config, indent=4)}")

    return config


def load_dataset(config):
    """ Dataset """
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = config["dataset"]["batch_size"]

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader


def setup_model():
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

    model = NeuralNetwork().to(best_available_device)
    print(model)

    return model, best_available_device


def save_model(model, config):
    model_filename = config["model_filename"]
    torch.save(model, model_filename)
    print(f"Model saved to {model_filename}")


""" training/testing """


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    # switch to train mode
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


""" Running the training and test loops """
if __name__ == "__main__":

    config = load_config()
    train_dataloader, test_dataloader = load_dataset(config)
    model, device = setup_model()

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = config["optimizer"]["learning_rate"]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = config["hyperparameters"]["epochs"]
    for t in range(epochs):
        print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    """ Save model"""
    save_model(model, config)
