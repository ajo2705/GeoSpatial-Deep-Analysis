import os
import importlib
import copy
import h5py
import numpy as np
import torch.nn as nn

from libraries.constants import ConfigParams, Base
from libraries.common_util import make_bool
from loss_landscape import net_manipulate
from loss_landscape.errors import InvalidModelException
from loss_landscape.util import load_none_on_error, Models
from loss_landscape.model_loader import load
from loss_landscape.net_manipulate import get_weights
from loss_landscape.scheduler import get_job_indices
from loss_landscape import evaluation

class TrainedModels:
    def __init__(self, config_manager, device):
        self.config_manager = config_manager
        self.model_file_1 = ""
        self.model_file_2 = ""
        self.model_file_3 = ""

        self.model_class = None
        self.device = device

        self._initialise_config_parameters()
        self.model_mapper = self._setup_model_mappers()

    def _setup_model_mappers(self):
        return {
            Models.Model1: self.model_file_1,
            Models.Model2: self.model_file_2,
            Models.Model3: self.model_file_3
        }

    @staticmethod
    def __wrap_file_abs_path(model_file):
        if model_file:
            return os.path.join(Base.BASE_MODEL_STORE_PATH, model_file)

    def _initialise_config_parameters(self):
        self.model_file_1 = self.__wrap_file_abs_path(self.load_config_val(ConfigParams.file1))
        self.model_file_2 = self.__wrap_file_abs_path(self.load_config_val(ConfigParams.file2))
        self.model_file_3 = self.__wrap_file_abs_path(self.load_config_val(ConfigParams.file3))

        module_name = self.load_config_val(ConfigParams.model_file)
        class_name = self.load_config_val(ConfigParams.model_class)

        module = importlib.import_module(module_name)
        self.model_class = getattr(module, class_name)

    @load_none_on_error
    def load_config_val(self, config_param):
        return self.config_manager.get_config_parameter(config_param)

    def get_x_dir_name_from_model_names(self):
        x_dir_file = ""
        if self.model_file_2 and os.path.exists(self.model_file_2):
            if self.model_file_1[:self.model_file_1.rfind('/')] == self.model_file_2[:self.model_file_2.rfind('/')]:
                # model_file and model_file2 are under the same folder
                x_dir_file += self.model_file_1 + '_' + self.model_file_2[self.model_file_2.rfind('/') + 1:]
            else:
                # model_file and model_file2 are under different folders
                prefix = os.path.commonprefix([self.model_file_1, self.model_file_2])
                prefix = prefix[0:prefix.rfind('/')]
                x_dir_file += self.model_file_1[:self.model_file_1.rfind('/')] + '_' + \
                              self.model_file_1[self.model_file_1.rfind('/') + 1:] + '_' + \
                              self.model_file_2[len(prefix) + 1: self.model_file_2.rfind('/')] + '_' + \
                              self.model_file_2[self.model_file_2.rfind('/') + 1:]
        else:
            x_dir_file += self.model_file_1

        return x_dir_file

    def get_y_dir_name_from_model_names(self):
        y_dir_file = ""
        if self.model_file_3 and os.path.exists(self.model_file_3):
            y_dir_file += self.model_file_3

        return y_dir_file

    def check_model_exist(self, model_type):
        return make_bool(self.model_mapper.get(model_type))

    def get_model_and_parameters(self, model_type):
        model_file = self.model_mapper.get(model_type)
        if model_file:
            net = load(self.model_class, self.device, model_file)
            weight = get_weights(net)
            state = copy.deepcopy(net.state_dict())
        else:
            raise InvalidModelException()

        return net, weight, state

    def load_surface_file_loss(self, surf_file, surf_file_name, dir_type, direction, dataloader, acc_key="train_acc", loss_key="train_loss"):
        xcoordinates = surf_file['xcoordinates'][:]
        ycoordinates = surf_file['ycoordinates'][:] if 'ycoordinates' in surf_file.keys() else None

        if loss_key not in surf_file.keys():
            shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
            losses = -np.ones(shape=shape)
            accuracies = -np.ones(shape=shape)
            surf_file[loss_key] = losses
            surf_file[acc_key] = accuracies
        else:
            losses = surf_file[loss_key][:]
            accuracies = surf_file[acc_key][:]
        surf_file.close()

        net, weight, state = self.get_model_and_parameters(Models.Model1)

        inds, coords = get_job_indices(losses, xcoordinates, ycoordinates)
        criterion = nn.MSELoss()

        print("Number of indices : ",len(inds))

        for count, ind in enumerate(inds):
            # Get the coordinates of the loss value being calculated
            coord = coords[count]

            if dir_type == 'weights':
                net_manipulate.set_weights(net, weight, direction, coord)
            elif dir_type == 'states':
                net_manipulate.set_states(net, state, direction, coord)

            loss, acc = evaluation.eval_loss(self.device, net, criterion, dataloader)
            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            # Periodically write to file, and always write after last update
            if count % 1000 == 0 or count == len(inds) - 1:
                print('Writing to file')
                f = h5py.File(surf_file_name, 'r+')
                f[loss_key][losses != -1] = losses[losses != -1]
                f[acc_key][accuracies != -1] = accuracies[accuracies != -1]
                f.flush()
                f.close()

        surf_file.close()