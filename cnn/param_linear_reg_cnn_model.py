import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ParameterizedCNN(nn.Module):
    def __init__(self, input_channels, hidden_size, kernel_size):
        super(ParameterizedCNN, self).__init__()

        # Replace input_channels with number of input channels created in patch after raster processing
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=kernel_size, stride=1, padding=1, bias=True)
        self.relu1 = nn.LeakyReLU()
        self.conv1_1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=1, bias=True)
        self.relu1_1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=kernel_size, stride=1, padding=1, bias=True)
        self.relu2 = nn.LeakyReLU()
        self.conv2_2 = nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=kernel_size, stride=1, padding=1, bias=True)
        self.relu2_2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        output_size = self._compute_fc1_size()
        self.fc1 = nn.Linear(output_size, 16)
        self.relu3 = nn.LeakyReLU()

        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.25)

    def _compute_fc1_size(self):
        #TODO: Hard coded --> account for variability

        x = torch.randn(3, 57, 10, 10)
        x = self._convolution_forward_pass(x)

        return x.size(1) * x.size(2) * x.size(3)

    def _convolution_forward_pass(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)

        return x

    def forward(self, x):
        # Forward pass through the layers
        x = self._convolution_forward_pass(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return np.squeeze(x)
