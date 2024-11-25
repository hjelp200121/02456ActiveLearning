import numpy as np
import torch

accuracies_committee = [[] for _ in range(5)]
for i in range(1,6):
    num_models = 2*i
    for j in range(10):
        accuracies_committee[i-1].append(j)

print(accuracies_committee)