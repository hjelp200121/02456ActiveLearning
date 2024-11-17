import torch
import torch.utils
import torch.utils.data

class UniformRandom:

    def __init__(self, size):
        self.size = size

    def select_subset(self, dataset):
        labelled_indices = torch.randperm(len(dataset))[:self.size]
        return torch.utils.data.Subset(dataset, labelled_indices)



