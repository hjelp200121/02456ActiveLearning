import torch
import numpy as np

from sklearn.cluster import AgglomerativeClustering
import torch.utils
import torch.utils.data

def plot_clustering_times(dataset, nmin=10, nmax=10000):
    import time
    import matplotlib.pyplot as plt

    sizes = np.linspace(nmin, nmax, 20, dtype=np.int64)
    times = []

    for size in sizes:
        t0 = time.time()
        subset = torch.utils.data.Subset(dataset, np.arange(size))
        clustering = AgglomerativeClustering(n_clusters=10).fit([x.flatten() for x, _ in subset])   
        t1 = time.time()

        times.append(t1 - t0)
        print(f"{size:05d}:\t{times[-1]}s")

    plt.plot(sizes, times)
    plt.savefig("figs/clustering_times.pdf")

class InputStore:

    def __init__(self):
        self.values = []

    def __call__(self, module, args, output):
        self.values.append(args[0])

class ClusterMargin:

    def __init__(self, model, train_fn, device, seed_sample_size=20, cluster_sample_size=80, margin_sample_size=80):
        self.model = model
        self.train_fn = train_fn
        self.device = device

        self.seed_sample_size = seed_sample_size
        self.cluster_sample_size = cluster_sample_size
        self.margin_sample_size = margin_sample_size
    
    def select_subset(self, dataset):
        
        # uniformly select seed_size datapoints to label 
        seed_sample = torch.randperm(len(dataset))[:self.seed_sample_size]
        
        # train on seed
        self.train_fn(self.model, torch.utils.data.Subset(dataset, seed_sample), self.device)
        
        # compute margin scores and clustering
        margin_scores, embeddings = self._compute_margin_scores_and_embeddings(dataset)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=8.0).fit(embeddings.cpu()) # TODO: less troll distance threshold or use n_clusters
        
        _, margin_sample = margin_scores.cpu().topk(self.margin_sample_size, largest=False) # TODO: filter out samples that are in the seed

        sampled_clusters = torch.tensor(clustering.labels_)[margin_sample].unique()
        sampled_cluster_sizes = torch.empty_like(sampled_clusters, dtype=torch.int64)

        for i, c in enumerate(sampled_clusters):
            sampled_cluster_sizes[i] = (clustering.labels_ == c).sum()

        sampled_cluster_sizes, key = sampled_cluster_sizes.sort()
        sampled_clusters = sampled_clusters[key]

        cluster_indices = [torch.where(clustering.labels_ == c)[0].tolist() for c, size in zip(sampled_clusters, sampled_cluster_sizes)]
        
        j = 0
        cluster_sample = []
        while len(cluster_sample) + 1 < self.cluster_sample_size:
            c = sampled_clusters[j]
            
            try:
                index = cluster_indices[j].pop()
                cluster_sample.append(index)
            except IndexError:
                pass
            
            j = (j + 1) % sampled_clusters.size()[0]

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
    import torchvision
    import torch
    import torch.nn as nn
    
    def train(model, train_set, device):


        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        loader = torch.utils.data.DataLoader(train_set, 32, shuffle=False, drop_last=False, num_workers=3)

        model.train()
        for image, target in iter(loader):
            image, target = image.to(device), target.to(device)

            output = model(image)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.to(device)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_set = torch.utils.data.Subset(train_set, [i for i in range(10000)])
    
    subset = ClusterMargin(model, train, device, seed_sample_size=200, cluster_sample_size=800, margin_sample_size=1200).select_subset(train_set)





