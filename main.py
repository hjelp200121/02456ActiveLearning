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





if __name__ == "__main__":
    plot_accuracies()

    


    


