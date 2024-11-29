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
from cluster_margin import ClusterMargin

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
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_set = torch.utils.data.Subset(train_set, [i for i in range(10000)])

    def objective(trial):

        seed_sample_frac = trial.suggest_float('seed_sample_frac', 0.1, 0.4)
        suggestions_per_sample= trial.suggest_float('suggestions_per_sample', 1.5, 10.0)
        cluster_count = trial.suggest_int('cluster_count', 10, 100, step=5)

        seed_all(1234)

        model = create_model().to(device)

        subset = ClusterMargin(model, device, 1000, seed_sample_frac, suggestions_per_sample, cluster_count).select_subset(train_set)

        train(model, subset, device)
        accuracy = test(model, test_set, device)

        return accuracy

    study_name = "cluster_margin_parameters"
    storage = f"sqlite:///{study_name}.db"

    study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage, load_if_exists=True)
    study.optimize(objective, timeout=9*60*60, show_progress_bar=False)
