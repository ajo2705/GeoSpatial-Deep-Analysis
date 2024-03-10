import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNeuralNetwork(nn.Module):
    def __init__(self, config):
        super(DeepNeuralNetwork, self).__init__()
        input_size = int(config['input_size'])
        output_size = config['output_size']
        hidden_sizes = list(map(int, config['hidden_size']))

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
