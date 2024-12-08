import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
from matplotlib.colors import Normalize

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import create_model
from uniform_random import UniformRandom


def validate(model, val_loader, criterion, device):
    loss_sum = 0.0
    count = 0
    
    model.eval()
    for image, target in iter(val_loader):
        image, target = image.to(device), target.to(device)
        output = model(image)

        loss = criterion(output, target)
        loss_sum += loss.item() * image.size(0)
        count += image.size(0)
    model.train()

    return loss_sum / count

def generate_losses_per_step():

    device = torch.device('cuda')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    val_size = int(0.1 * len(train_set))
    indices = torch.randperm(len(train_set))
    val_set = torch.utils.data.Subset(train_set, indices[:val_size])
    train_set = torch.utils.data.Subset(train_set, indices[val_size:])
    
    num_graphs = 8
    subset_sizes = [256 + 256*3*i for i in range(num_graphs-1)]
    subset_sizes.append(20*256)

    num_steps = np.arange(0, 425, 25, dtype=np.int32)
    
    val_loader = DataLoader(val_set, 32, shuffle=False, drop_last=False, num_workers=3)

    losses = []

    for size in tqdm(subset_sizes):
        subset = UniformRandom(size).select_subset(train_set)

        train_loader = DataLoader(subset, 32, shuffle=True, drop_last=False, num_workers=3)

        model = create_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        step = 0
        size_losses = []
        
        is_training = True

        while is_training:

            for image, target in iter(train_loader):

                if step in num_steps:
                    size_losses.append(validate(model, val_loader, criterion, device))
                
                if step == num_steps[-1]:
                    is_training = False
                    break
                
                image, target = image.to(device), target.to(device)
                output = model(image)

                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                step += 1
        
        losses.append(size_losses)

    torch.save(torch.tensor(losses), "results/losses_per_step.pt")

def plot_losses_per_step():
    losses = torch.load("results/losses_per_step.pt", weights_only=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_ylabel("Loss", fontsize=12)
    ax.set_xlabel("Number of steps", fontsize=12)
    
    colormap = cm.viridis

    num_graphs = 8

    subset_sizes = [256 + 256*3*i for i in range(num_graphs-1)]
    subset_sizes.append(20*256)

    num_steps = np.arange(0, 425, 25, dtype=np.int32)

    #Normalize the color range for the number of graphs
    norm = Normalize(vmin=np.min(subset_sizes), vmax=np.max(subset_sizes))
    colors = colormap(np.linspace(0, 1, num_graphs))
    
    for i, (subset_size, color) in enumerate(zip(subset_sizes, colors)):

        size_losses_np = losses[i,:].numpy()
        ax.plot(num_steps, size_losses_np, label=f"Size={subset_size}", color=color)

    plt.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([]) 
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Subset size", fontsize=12)

    plt.savefig("figs/losses_per_step.pdf")



if __name__ == "__main__":

    generate_losses_per_step()
    plot_losses_per_step()

