import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, devices: list):
        super(MLP, self).__init__()
        self.devices = devices
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(self.devices[0])
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(self.devices[0])
        self.fc3 = nn.Linear(hidden_dim, output_dim).to(self.devices[1])
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = nn.Flatten()(x_in.to(self.devices[0])).to(self.devices[0])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x.to(self.devices[1]))
        out = self.relu(x)
        return out
