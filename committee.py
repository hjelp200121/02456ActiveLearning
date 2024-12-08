import torch
import torch.utils
import torch.utils.data
from model import train, test

class Committee:

    def __init__(self, models, device, use_soft=True, seed_sample_size=20, vote_size=80):
        self.models = models
        
        self.device = device
        self.use_soft = use_soft

        self.seed_sample_size = seed_sample_size
        self.vote_size = vote_size
    
    def query_hard(self, train_set, num_classes=10, batch_size=32):
        #Using normalized vote entropy to quantify the disagreement between the models on all the samples in the training set
        with torch.no_grad():
            l = len(self.models)
            loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, drop_last=False, num_workers=3)

            for i in range(l):
                self.models[i].eval()

            #Create disagreement Array
            disagreement_array = torch.empty(len(train_set),device=self.device)

            for i, (image, _) in enumerate(iter(loader)):
                image = image.to(self.device)
                prediction = [None]*l
                for j in range(l):
                    prediction[j] = self.models[j](image).softmax(dim=1).argmax(dim=1)
                arr = torch.zeros(prediction[0].size()).to(self.device)
                stack = torch.stack(prediction)

                for j in range(num_classes):
                    Vc = (stack == j).sum(axis=0)
                    arr += Vc/l*torch.log(Vc.clamp(min=1)/l)

                disagreement_array[i*batch_size:i*batch_size+prediction[0].size(dim=0)] = -arr/torch.log(torch.tensor(l))

            return disagreement_array
        
    def query_soft(self, train_set, num_classes=10, batch_size=32):
        with torch.no_grad():
            l = len(self.models)
            loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, drop_last=False, num_workers=3)

            for i in range(l):
                self.models[i].eval()

            #Create disagreement Array
            disagreement_array = torch.empty(len(train_set), device=self.device)

            for i, (image, _) in enumerate(iter(loader)):
                image = image.to(self.device)
                prediction = torch.zeros((len(image),num_classes)).to(self.device)
                for j in range(l):
                    prediction += self.models[j](image).softmax(dim=1)
                prediction = prediction/l

                Hp = torch.zeros(len(image)).to(self.device)
                for j in range(num_classes):
                    Hp -= prediction[:,j]*torch.log(prediction[:,j].clamp(min=1e-9))

                disagreement_array[i*batch_size:i*batch_size+prediction.size(dim=0)] = Hp

            return disagreement_array

    def select_subset(self, dataset):

        # uniformly select seed_size datapoints to label 
        seed_sample = torch.randperm(len(dataset))[:self.seed_sample_size]
        subset = torch.utils.data.Subset(dataset, seed_sample)
        
        # train models on seed
        for i in range(len(self.models)):
            train(self.models[i], subset, self.device)
        
        #Use a metric to determine how much the models agree
        if self.use_soft:
            disagreementMatrix = self.query_soft(dataset, 10, 32)
        else:
            disagreementMatrix = self.query_hard(dataset, 10, 32)
        
        _, vote_sample = disagreementMatrix.topk(self.vote_size, largest=True)
        sample = torch.concat([seed_sample, vote_sample.cpu()])
        sub = torch.utils.data.Subset(dataset, sample)

        return sub

if __name__ == "__main__":
    import torchvision
    import torch.nn as nn
    from tqdm import tqdm

    from model import create_model
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    size = 1024
    seed_sample_size = 256
    vote_size = size - seed_sample_size
    

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    perm = torch.randperm(len(train_set))
    val_size = 10000
    val_set = torch.utils.data.Subset(train_set, perm[:val_size])
    train_set = torch.utils.data.Subset(train_set, perm[val_size:])

    for num_models in range(2, 10):

        accuracies = []

        for _ in tqdm(range(10)):
            model = create_model().to(device)
            models = [create_model().to(device) for i in range(num_models)]
            
            subset = Committee(models, device, False, seed_sample_size, vote_size).select_subset(train_set)
            train(model, subset, device)

            accuracy = test(model, val_set, device)
            accuracies.append(accuracy)
        
        print(torch.tensor(accuracies).mean())
