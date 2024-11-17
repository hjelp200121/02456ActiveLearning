import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from cluster_margin import ClusterMargin
from uniform_random import UniformRandom

def train(model, train_set, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(train_set, 32, shuffle=False, drop_last=False, num_workers=3)

    model.train()
    for image, target in iter(loader):
        image, target = image.to(device), target.to(device)

        output = model(image)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, test_set, device):

    loader = torch.utils.data.DataLoader(test_set, 32, shuffle=False, drop_last=False, num_workers=3)

    model.eval()

    correct = 0
    total = 0

    for image, target in iter(loader):
        image, target = image.to(device), target.to(device)
        output = model(image).softmax(dim=1)

        prediction = output.argmax(dim=1)

        correct += (prediction == target).sum().item()
        total += image.size()[0]

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    import torchvision

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.to(device)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    val_size = int(0.1 * len(train_set))
    indices = torch.randperm(len(train_set))
    val_set = torch.utils.data.Subset(train_set, indices[:val_size])
    train_set = torch.utils.data.Subset(train_set, indices[val_size:])

    subset_sizes = np.linspace(100, 10000, 20, dtype=np.int32)
    accuracies = []

    for size in subset_sizes:
        subset = UniformRandom(size).select_subset(train_set)

        model_copy = deepcopy(model) 

        train(model_copy, subset, device)
        accuracy = test(model_copy,  test_set, device)

        accuracies.append(accuracy)
    
    plt.plot(subset_sizes, accuracies)
    plt.savefig("figs/test.pdf")

    


    


