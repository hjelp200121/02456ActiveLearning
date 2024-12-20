



import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torchvision

import time

from model import create_model, train, test

class Margin:

    def __init__(self, model, device, seed_sample_size, margin_sample_size):
        self.model = model
        self.device = device

        self.seed_sample_size = seed_sample_size
        self.margin_sample_size = margin_sample_size

    def select_subset(self, dataset):
        
        # uniformly select seed_size datapoints to label
        perm = torch.randperm(len(dataset))
        
        seed_sample = perm[:self.seed_sample_size]
        not_seed_sample = perm[self.seed_sample_size:]

        # train on seed
        train(self.model, torch.utils.data.Subset(dataset, seed_sample), self.device)
        
        # compute margin scores
        margin_scores = self._compute_margin_scores(dataset)

        _, margin_sample = margin_scores[not_seed_sample].cpu().topk(self.margin_sample_size, largest=False)
        margin_sample = not_seed_sample[margin_sample] # remap from not seed pool to entire dataset

        sample = torch.concat([seed_sample, margin_sample])
        
        return torch.utils.data.Subset(dataset, sample)


    def _compute_margin_scores(self, dataset):
        self.model.eval()
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(dataset, 32, shuffle=False, drop_last=False, num_workers=3)
            
            margin_scores = []

            for image, _ in iter(loader):
                output = self.model(image.to(self.device)).softmax(dim=1)

                top2, _ = output.topk(2, dim=1, sorted=True)
                margin_scores.append(top2[:,0] - top2[:,1])
            
            margin_scores = torch.concat(margin_scores)

        return margin_scores



    
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_set = torch.utils.data.Subset(train_set, [i for i in range(10000)])
    
    torch.manual_seed(1234)

    model = create_model().to(device)

    t0 = time.time()
    subset = Margin(model, device, 192, 1024 - 192).select_subset(train_set)
    t1 = time.time()

    model = create_model().to(device)

    train(model, subset, device)
    accuracy = test(model, test_set, device)
    t = t1 - t0

    print(f"accuracy: {accuracy}, time: {t}")
    





