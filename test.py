import numpy as np
import torch

disagreementArray = torch.zeros(10000)

for i in range(312):
    prediction1 = (torch.rand(32)*10).type(torch.uint8)
    prediction2 = (torch.rand(32)*10).type(torch.uint8)
    prediction3 = (torch.rand(32)*10).type(torch.uint8)
    prediction4 = (torch.rand(32)*10).type(torch.uint8)
    arr = torch.zeros(prediction1.size())
    stack = torch.stack([prediction1, prediction2, prediction3, prediction4])

    for j in range(10):
        Vc = (stack == j).sum(axis=0)
        arr += Vc/4*np.log(Vc/4, where=Vc!=0)
    disagreementArray[i*prediction1.size(dim=0):(i+1)*prediction1.size(dim=0)] = -arr/np.log(4)
print(disagreementArray)