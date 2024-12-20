import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST dataset
transform = transforms.ToTensor()
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Select 64 images
images = [mnist[i][0] for i in range(64)]  # Extract only the images

# Create a grid of 64 images
fig, axes = plt.subplots(8, 8, figsize=(8, 8), dpi=300)

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap='gray')  # Remove extra dimension with squeeze()
    ax.axis('off')  # Remove axes for a clean grid

# Remove whitespace around the grid
plt.subplots_adjust(wspace=0, hspace=0)

# Save the figure to a file
plt.savefig("results/mnist_grid_64.pdf", bbox_inches='tight', pad_inches=0)
