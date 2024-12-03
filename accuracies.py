
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
from committee import Committee

def split_whole_batches(size, frac):
    x = int(round(frac * size / 32)) * 32

    return x, size - x


def select_uniform_random(device, dataset, size):
    return UniformRandom(size).select_subset(dataset)

def select_cluster_margin(device, dataset, size):
    
    model = create_model().to(device)

    # params found by optuna
    seed_sample_frac = 0.29556213082513294
    cluster_count=35
    suggestions_per_sample=2.0174904696016838

    # split at nearest 32 to avoid having partial batches
    seed_sample_size, cm_sample_size = split_whole_batches(size, seed_sample_frac) 
    suggestion_size = int(suggestions_per_sample * cm_sample_size)

    cm = ClusterMargin(model, device, seed_sample_size, cm_sample_size, suggestion_size, cluster_count)

    return  cm.select_subset(dataset)

def select_committee(device, dataset, size):
    num_models = 4

    # split at nearest 32 to avoid having partial batches
    seed_sample_size, vote_size = split_whole_batches(size, 0.2) 
       
    models = [create_model().to(device) for i in range(num_models)]
    
    return Committee(models, device, False, seed_sample_size, vote_size).select_subset(dataset)


def generate_accuracies(select_fn, name):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    subset_sizes = [256 * i for i in range(1, 21)]
    
    accuracies = []

    for size in tqdm(subset_sizes):
        model = create_model().to(device)

        subset = select_fn(device, train_set, size)
        
        train(model, subset, device)
        accuracy = test(model,  test_set, device)

        accuracies.append(accuracy)

    torch.save(torch.tensor(accuracies), f"results/accuracies_{name}.pt")

def load_accuracies(name):
    accuracies = torch.stack([torch.load(f"results/accuracies_{name}_{i}.pt", weights_only=True) for i in range(1)])
    return accuracies.mean(dim=0), accuracies.std(dim=0)

def plot_accuracies():
    plt.ylabel("Accuracy")
    plt.xlabel("Number of labelled points")

    names = ["uniform_random", "cluster_margin", "committee_soft", "committee_hard"]
    labels = ["Uniform", "Cluster-Margin", "Committee (Soft)", "Committee (Hard)"]

    subset_sizes = [256 * i for i in range(1, 21)]

    for name, label in zip(names, labels):
        mean, std = load_accuracies(name)

        plt.errorbar(subset_sizes, mean, std, label=label)

    plt.legend()

    plt.savefig("figs/accuracy.pdf")



if __name__ == "__main__":

    torch.manual_seed(1234)

    # for i in range(0, 1):
    #     generate_accuracies(select_committee, f"committee_hard_{i}")

    plot_accuracies()


