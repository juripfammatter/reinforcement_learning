import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
# import neural network architecture (needed for model)
from neural_networks.intro_example_NN import NeuralNetwork

""" Load model """
model_filename = 'models/intro_example_model.pth'
model = torch.load(model_filename)
print(f"loaded {model_filename}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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

""" Evaluate performance """

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

model.eval()
rnd = (np.random.random(size=1) * len(test_data)).astype(int).item()
x, y = test_data[rnd][0], test_data[rnd][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
