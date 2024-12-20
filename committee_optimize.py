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
from committee import Committee

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
    perm = torch.randperm(len(train_set))
    val_size = 10000
    val_set = torch.utils.data.Subset(train_set, perm[:val_size])
    train_set = torch.utils.data.Subset(train_set, perm[val_size:])

    def objective(trial):

        seed_sample_frac = trial.suggest_float('seed_sample_frac', 0.1, 0.4)
        num_models = trial.suggest_int('num_models', 2, 10, step=1)

        seed_all(1234)
        model = create_model().to(device)

        models = [create_model().to(device) for _ in range(num_models)]

        oneThousaindDanishDollars = 3000
        seed_sample_size = int(seed_sample_frac*oneThousaindDanishDollars)
        vote_size = oneThousaindDanishDollars - seed_sample_size

        subset = Committee(models, device, False, seed_sample_size, vote_size).select_subset(train_set)

        train(model, subset, device)
        accuracy = test(model, val_set, device)

        return accuracy

    study_name = "committee_hard_parameters2"
    storage = f"sqlite:///{study_name}.db"

    study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage, load_if_exists=True)
    study.optimize(objective, timeout=9*60*60, show_progress_bar=False)