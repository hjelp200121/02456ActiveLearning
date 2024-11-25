import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from model import create_model, train, test
from cluster_margin import ClusterMargin
from uniform_random import UniformRandom



def plot_accuracies():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model().to(device)

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

    


    


