import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from cluster_margin import ClusterMargin
from uniform_random import UniformRandom

def train(model, train_set, device):


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(train_set, 32, shuffle=False, drop_last=True, num_workers=3)

    model.train()
    for image, target in iter(loader):
        image, target = image.to(device), target.to(device)

        output = model(image)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, test_set, device):

    loader = torch.utils.data.DataLoader(test_set, 32, shuffle=False, drop_last=True, num_workers=3)

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

def plot_accuracies():
    
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

    if True:
        train_set = torch.utils.data.Subset(train_set, [i for i in range(10000)])

    val_size = int(0.1 * len(train_set))
    indices = torch.randperm(len(train_set))
    val_set = torch.utils.data.Subset(train_set, indices[:val_size])
    train_set = torch.utils.data.Subset(train_set, indices[val_size:])

    subset_sizes = np.linspace(100, 5000, 20, dtype=np.int32)
    accuracies_uniform = []
    accuracies_cluster_margin = []

    for size in tqdm(subset_sizes):
        subset = UniformRandom(size).select_subset(train_set)

        model_copy = deepcopy(model) 

        train(model_copy, subset, device)
        accuracy = test(model_copy,  test_set, device)

        accuracies_uniform.append(accuracy)
    
    for size in tqdm(subset_sizes):

        seed_sample_size = int(0.2 * size)
        cluster_sample_size = size - seed_sample_size
        margin_sample_size = int(1.5 * cluster_sample_size)

        subset = ClusterMargin(deepcopy(model), train, device, seed_sample_size, cluster_sample_size, margin_sample_size).select_subset(train_set)

        model_copy = deepcopy(model)
        train(model_copy, subset, device)
        accuracy = test(model_copy, test_set, device, )

        accuracies_cluster_margin.append(accuracy)

    plt.ylabel("Accuracy")
    plt.xlabel("Number of labelled points")
    plt.ylim(0.0, 1.0)

    plt.plot(subset_sizes, accuracies_uniform, label="Uniform")
    plt.plot(subset_sizes, accuracies_cluster_margin, label="Cluster-Margin")

    plt.legend()

    plt.savefig("figs/accuracy.pdf")


if __name__ == "__main__":
    plot_accuracies()

    


    


