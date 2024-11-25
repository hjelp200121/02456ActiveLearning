import torch
import torch.utils
import torch.utils.data

class Committee:

    def __init__(self, models, train_fn, device, seed_sample_size=20, vote_size=80):
        self.models = models
        
        self.train_fn = train_fn
        self.device = device

        self.seed_sample_size = seed_sample_size
        self.vote_size = vote_size
    
    def query(self, train_set, num_classes=10, batch_size=32):
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, drop_last=False, num_workers=3)

            for i in range(len(self.models)):
                self.models[i].eval()

            #Create disagreement Array
            disagreement_array = torch.empty(len(train_set))

            for i, (image, _) in enumerate(iter(loader)):
                image = image.to(self.device)
                prediction = [None]*len(self.models)
                for i in range(len(self.models)):
                    prediction[i] = self.models[i](image).softmax(dim=1).argmax(dim=1)
                arr = torch.zeros(prediction[0].size()).to(self.device)
                stack = torch.stack(prediction)

                for j in range(num_classes):
                    Vc = (stack == j).sum(axis=0)
                    arr += Vc/4*torch.log(Vc.clamp(min=1)/4)

                disagreement_array[i*batch_size:i*batch_size+prediction[0].size(dim=0)] = -arr/torch.log(torch.tensor(4))

            return disagreement_array

    def select_subset(self, dataset):
        
        # uniformly select seed_size datapoints to label 
        seed_sample = torch.randperm(len(dataset))[:self.seed_sample_size]
        subset = torch.utils.data.Subset(dataset, seed_sample)
        
        # train models on seed
        for i in range(len(self.models)):
            self.train_fn(self.models[i], subset, self.device)
        
        #Using normalized vote entropy to quantify the disagreement between the models on all the samples in the training set
        disagreementMatrix = self.query(dataset, 10, 32)

        _, vote_sample = disagreementMatrix.topk(self.vote_size, largest=False)

        sample = torch.concat([seed_sample, vote_sample])
        return torch.utils.data.Subset(dataset, sample)

if __name__ == "__main__":
    import torchvision
    import torch
    import torch.nn as nn
    num_models = 4
    
    def train(model, train_set, device):

        num_steps = 400

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        loader = torch.utils.data.DataLoader(train_set, 32, shuffle=True, drop_last=False, num_workers=3)

        model.train()

        step = 0
        while True:
            for image, target in iter(loader):
                if image.size()[0] == 1:
                    continue
                
                image, target = image.to(device), target.to(device)

                output = model(image)

                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                step += 1

                if step >= num_steps:
                    return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = [None]*num_models
    for i in range(num_models):
        models[i] = torchvision.models.resnet18()
        models[i].fc = torch.nn.Linear(models[i].fc.in_features, 10)
        models[i].conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        models[i] = models[i].to(device)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_set = torch.utils.data.Subset(train_set, [i for i in range(10000)])
    
    subset = Committee(models, train, device, seed_sample_size=200, vote_size=800).select_subset(train_set)