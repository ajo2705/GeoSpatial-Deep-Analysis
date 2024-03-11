import torch
import torch.nn as nn


class ParameterizedCNN(nn.Module):
    def __init__(self, num_classes, hidden_size, kernel_size):
        super(ParameterizedCNN, self).__init__()

        # Replace input_channels with number of input channels created in patch after raster processing
        self.conv1 = nn.Conv2d(97, hidden_size, kernel_size=kernel_size, stride=1, padding=kernel_size - 1)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=kernel_size, stride=1, padding=kernel_size - 1)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        output_size = self._compute_fc1_size()
        self.fc1 = nn.Linear(output_size, 64)
        self.relu3 = nn.LeakyReLU()

        self.fc2 = nn.Linear(64, num_classes)

    def _compute_fc1_size(self):
        x = torch.randn(3, 97, 8, 8)
        x = self._convolution_forward_pass(x)

        return x.size(1) * x.size(2) * x.size(3)

    def _convolution_forward_pass(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        return x

    def forward(self, x):
        # Forward pass through the layers
        x = self._convolution_forward_pass(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)

        return x
