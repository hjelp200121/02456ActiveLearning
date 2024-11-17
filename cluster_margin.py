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

    def __init__(self, model, train_fn, device, seed_size=100):
        self.model = model
        self.train_fn = train_fn
        self.device = device

        self.seed_size = seed_size
    
    def select_subset(self, dataset):
        
        # uniformly select seed_size datapoints to label 
        labelled_indices = torch.randperm(len(dataset))[:self.seed_size]
        
        # train on seed
        self.train_fn(self.model, torch.utils.data.Subset(dataset, labelled_indices), self.device)
        
        # compute margin scores and clustering
        margin_scores, embeddings = self._compute_margin_scores_and_embeddings(dataset)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=10.0).fit(embeddings.cpu()) # less troll distance threshold or use n_clusters
        


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






