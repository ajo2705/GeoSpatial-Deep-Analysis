import torch.nn as nn


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=None):
        super(SkipBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.pool = None

        if pooling:
            if pooling['type'] == 'max':
                self.pool = nn.MaxPool2d(kernel_size=pooling['kernel_size'], stride=pooling['stride'])
            elif pooling['type'] == 'avg':
                self.pool = nn.AvgPool2d(kernel_size=pooling['kernel_size'], stride=pooling['stride'])


        if in_channels != out_channels or stride != 1:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.skip_batch_norm = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_batch_norm(identity)

        out += identity  # Element-wise addition

        if self.pool:
            out = self.pool(out)

        out = self.relu(out)  # Apply activation after merging

        return out