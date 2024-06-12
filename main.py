import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import MLP
from train import training_loop


training_data = datasets.MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

model = MLP(input_dim=28*28, hidden_dim=64, output_dim=10)
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 100

training_loop(model=model, train_loader=train_loader, epoch=num_epoch, lossFn=lossFn, optimizer=optimizer)



