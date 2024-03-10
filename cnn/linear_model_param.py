import torch
import torch.nn as nn
import numpy as np

from cnn.ConvBlock import ConvBlock
from cnn.SkipBlock import SkipBlock


class ParameterizedCNN(nn.Module):
    def __init__(self, config):
        super(ParameterizedCNN, self).__init__()

        self.layers = nn.ModuleList()
        self.fc_layers = nn.Sequential(*self._make_layers(config))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layers(self, config):
        in_channels = config['input_channels']

        nn_block = ConvBlock

        if config.get('use_skip_connection', False):
            nn_block = SkipBlock

        for layer_config in config['layers']:
            self.layers.append(nn_block(in_channels, **layer_config))
            in_channels = layer_config['out_channels']

        fc_layers = []
        fc_input = in_channels * config['adaptive_pooling']['output_size'][0] * \
                   config['adaptive_pooling']['output_size'][1]
        for fc_out in config['fully_connected']:
            fc_layers.append(nn.Linear(fc_input, fc_out))
            fc_input = fc_out

        return fc_layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        for fc in self.fc_layers:
            x = fc(x)

        return np.squeeze(x)
