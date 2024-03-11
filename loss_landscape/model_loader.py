import os
import torch


def load(model_class, device, model_file=None):
    net = model_class().to(device)

    if model_file and os.path.exists(model_file):
        state_model = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in state_model.keys():
            net.load_state_dict(state_model['state_dict'])
        else:
            net.load_state_dict(state_model)

    net.eval()
    return net
