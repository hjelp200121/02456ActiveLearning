
import torch
import torchvision
import torch.nn as nn

def create_model(seed=1234):
    torch.manual_seed(seed)

    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return model

def train(model, train_set, device):

    num_steps = 350

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

def test(model, test_set, device):

    loader = torch.utils.data.DataLoader(test_set, 32, shuffle=False, drop_last=False, num_workers=3)

    model.eval()

    correct = 0
    total = 0

    for image, target in iter(loader):
        image, target = image.to(device), target.to(device)
        output = model(image).softmax(dim=1)

        prediction = output.argmax(dim=1)

        correct += (prediction == target).sum().item()
        total += image.size()[0]

    accuracy = correct / total
    return accuracy



