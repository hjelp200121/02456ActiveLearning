import torch
import numpy as np
import torch.utils
import torch.utils.data

class Committee:

    def __init__(self, model1, model2, model3, model4, train_fn, device, seed_sample_size=20, vote_size=80):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        
        self.train_fn = train_fn
        self.device = device

        self.seed_sample_size = seed_sample_size
        self.vote_size = vote_size
    
    def query(self, train_set, num_classes=10, batch_size=32):
        loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, drop_last=False, num_workers=3)

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model4.eval()

        #Create disagreement Array
        disagreementArray = torch.empty(10000)

        for i, (image, _) in enumerate(iter(loader)):
            image = image.to(self.device)
            prediction1 = self.model1(image).softmax(dim=1).argmax(dim=1)
            prediction2 = self.model2(image).softmax(dim=1).argmax(dim=1)
            prediction3 = self.model3(image).softmax(dim=1).argmax(dim=1)
            prediction4 = self.model4(image).softmax(dim=1).argmax(dim=1)
            arr = torch.zeros(prediction1.size()).to(self.device)
            stack = torch.stack([prediction1, prediction2, prediction3, prediction4])

            for j in range(num_classes):
                Vc = (stack == j).sum(axis=0)
                arr += Vc/4*torch.log(Vc.clamp(min=1)/4)

            disagreementArray[i*batch_size:i*batch_size+prediction1.size(dim=0)] = -arr/torch.log(torch.tensor(4))

        return disagreementArray

    def select_subset(self, dataset):
        
        # uniformly select seed_size datapoints to label 
        seed_sample = torch.randperm(len(dataset))[:self.seed_sample_size]
        
        # train models on seed
        self.train_fn(self.model1, torch.utils.data.Subset(dataset, seed_sample), self.device)
        self.train_fn(self.model2, torch.utils.data.Subset(dataset, seed_sample), self.device)
        self.train_fn(self.model3, torch.utils.data.Subset(dataset, seed_sample), self.device)
        self.train_fn(self.model4, torch.utils.data.Subset(dataset, seed_sample), self.device)
        
        #Using normalized vote entropy to quantify the disagreement between the models on all the samples in the training set
        disagreementMatrix = self.query(dataset, 10, 32)

        _, vote_sample = disagreementMatrix.topk(self.vote_size, largest=False)

        sample = torch.concat([seed_sample, vote_sample])
        return torch.utils.data.Subset(dataset, sample)

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

    model1 = torchvision.models.resnet18()
    model1.fc = torch.nn.Linear(model1.fc.in_features, 10)
    model1.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model1 = model1.to(device)
    model2 = torchvision.models.resnet18()
    model2.fc = torch.nn.Linear(model2.fc.in_features, 10)
    model2.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model2 = model2.to(device)
    model3 = torchvision.models.resnet18()
    model3.fc = torch.nn.Linear(model3.fc.in_features, 10)
    model3.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model3 = model3.to(device)
    model4 = torchvision.models.resnet18()
    model4.fc = torch.nn.Linear(model4.fc.in_features, 10)
    model4.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model4 = model4.to(device)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_set = torch.utils.data.Subset(train_set, [i for i in range(10000)])
    
    subset = Committee(model1, model2, model3, model4, train, device, seed_sample_size=200, vote_size=800).select_subset(train_set)