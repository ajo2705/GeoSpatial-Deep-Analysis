import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.pool = None
        if pooling:
            if pooling['type'] == 'max':
                self.pool = nn.MaxPool2d(kernel_size=pooling['kernel_size'], stride=pooling['stride'])
            elif pooling['type'] == 'avg':
                self.pool = nn.AvgPool2d(kernel_size=pooling['kernel_size'], stride=pooling['stride'])

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        return x