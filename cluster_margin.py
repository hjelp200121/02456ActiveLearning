
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torchvision

import optuna
import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from model import create_model, train, test


class InputStore:

    def __init__(self):
        self.values = []

    def __call__(self, module, args, output):
        self.values.append(args[0])

def inv_perm(perm):
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(0, len(perm))
    return inv_perm

class ClusterMargin:

    def __init__(self, model, device, sample_size, seed_sample_frac, clusters_per_sample, cluster_count):
        self.model = model
        self.device = device

        self.seed_sample_size = int(seed_sample_frac * sample_size)
        self.cluster_sample_size = sample_size - self.seed_sample_size
        self.margin_sample_size = int(clusters_per_sample * self.cluster_sample_size)
        self.cluster_count = cluster_count

    def select_subset(self, dataset):
        
        # uniformly select seed_size datapoints to label
        perm = torch.randperm(len(dataset))
        
        seed_sample = perm[:self.seed_sample_size]
        not_seed_sample = perm[self.seed_sample_size:]

        # train on seed
        train(self.model, torch.utils.data.Subset(dataset, seed_sample), self.device)
        
        # compute margin scores and clustering
        margin_scores, embeddings = self._compute_margin_scores_and_embeddings(dataset)
        clustering = AgglomerativeClustering(n_clusters=self.cluster_count).fit(embeddings.cpu())

        _, margin_sample = margin_scores[not_seed_sample].cpu().topk(self.margin_sample_size, largest=False)
        margin_sample = not_seed_sample[margin_sample] # rema from not seed pool to entire dataset
        
        labels = torch.tensor(clustering.labels_)[margin_sample]

        clusters = [margin_sample[labels == l].tolist() for l in labels.unique()]
        clusters.sort(key=len)

        j = 0
        cluster_sample = []
        while len(cluster_sample) + 1 < self.cluster_sample_size:
            try:
                index = clusters[j].pop()
                cluster_sample.append(index)
            except IndexError:
                pass

            j = (j + 1) % len(labels.unique())

        sample = torch.concat([seed_sample, torch.tensor(cluster_sample, dtype=torch.int64)])

        return torch.utils.data.Subset(dataset, sample)


    def _compute_margin_scores_and_embeddings(self, dataset):
        self.model.eval()
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(dataset, 32, shuffle=False, drop_last=False, num_workers=3)
            
            margin_scores = []
            
            input_store = InputStore()
            hook_handle = self.model.fc.register_forward_hook(input_store) # need generic way of getting last layer?

            for image, _ in iter(loader):
                output = self.model(image.to(self.device)).softmax(dim=1)

                top2, _ = output.topk(2, dim=1, sorted=True)
                margin_scores.append(top2[:,0] - top2[:,1])
            
            hook_handle.remove()

            margin_scores = torch.concat(margin_scores)
            embeddings = torch.concat(input_store.values)
        
        return margin_scores, embeddings



    
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_set = torch.utils.data.Subset(train_set, [i for i in range(20000)])
    
    torch.manual_seed(1234)

    trials = 8

    seed_sample_fracs = [0.1, 0.2, 0.3, 0.4]
    clusters_per_samples = [1.5, 3.0, 5.0, 10.0]
    cluster_counts = [10, 40, 70, 100]

    accuracies = torch.empty([len(seed_sample_fracs), len(clusters_per_samples), len(cluster_counts), trials])
    times = torch.empty([len(seed_sample_fracs), len(clusters_per_samples), len(cluster_counts), trials])

    def run(seed_sample_frac, clusters_per_sample, cluster_count):
        print(f"seed_sample_frac: {seed_sample_frac:.1f}, clusters_per_sample: {clusters_per_sample:.1f}, cluster_count: {cluster_count}")

        accuracies = []
        times = []

        for trial in range(trials):
            model = create_model().to(device)

            t0 = time.time()
            subset = ClusterMargin(model, device, 1000, seed_sample_frac, clusters_per_sample, cluster_count).select_subset(train_set)
            t1 = time.time()

            train(model, subset, device)
            accuracies.append(test(model, test_set, device))
            times.append(t1 - t0)

            print(f"trial: {trial}, accuracy: {accuracies[-1]}, time: {times[-1]}")

        accuracies = torch.tensor(accuracies)
        times = torch.tensor(times)

        print(f"accuracy mean: {accuracies.mean():.4f}, accuracy std: {accuracies.std():.4f}, time mean: {times.mean():.4f}, time std: {times.std():.4f}")
        print()

        return accuracies, times

    # original:
    # accuracy mean: 0.9424, accuracy std: 0.0034, time mean: 24.4212, time std: 0.2470

    for i, seed_sample_frac in enumerate(seed_sample_fracs):
        for j, clusters_per_sample in enumerate(clusters_per_samples):
            for k, cluster_count in enumerate(cluster_counts):

                a, t = run(seed_sample_frac, clusters_per_sample, cluster_count)

                accuracies[i,j,k,:] = a
                times[i,j,k,:] = t

    torch.save(accuracies, "results/parameter_accuracies.pt")
    torch.save(times, "results/parameter_times.pt")
    





