import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import os
import optuna
import numpy as np

from model import create_model, train, test
from margin import Margin


import random

def make_deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    make_deterministic()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    val_set = torch.utils.data.Subset(train_set, [i for i in range(10000, 20000)])
    train_set = torch.utils.data.Subset(train_set, [i for i in range(10000)])

    subset_size = 1024

    def objective(trial):

        seed_sample_size = 32 * trial.suggest_int('seed_sample_batches', 1, 16)

        seed_all(1234)

        model = create_model().to(device)
        
        subset = Margin(model, device, seed_sample_size, subset_size - seed_sample_size).select_subset(train_set)

        train(model, subset, device)
        accuracy = test(model, val_set, device)
        
        return accuracy

    study_name = "margin_parameters"
    storage = f"sqlite:///{study_name}.db"
    search_space = {
        'seed_sample_batches': [i for i in range(1, 17)]
    }
    sampler = optuna.samplers.GridSampler(search_space)

    study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage, load_if_exists=True, sampler=sampler)
    study.optimize(objective, timeout=9*60*60, show_progress_bar=False)
