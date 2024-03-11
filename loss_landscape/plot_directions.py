import os
import h5py

from libraries.constants import ConfigParams
from libraries.common_util import make_bool
from loss_landscape.util import load_none_on_error, Models
from loss_landscape.net_manipulate import get_diff_states, get_diff_weights, get_random_states, get_random_weights
from loss_landscape.h5_util import write_list


class PlotDirections:
    def __init__(self, config_manager):
        """
              dir_type: 'weights' or 'states', type of directions.
              ignore: 'biasbn', ignore biases and BN parameters.
              norm: direction normalization method, including
                    'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        """
        self.config_manager = config_manager
        self.dir_file = ""
        self.dir_type = ""

        self.x_cords = ""
        self.x_ignore = None
        self.x_norm = None

        self.y_cords = ""
        self.y_ignore = None
        self.y_norm = None

        self.same_dir = False
        self.surf_file = ""

        self._initialise_config_parameters()

    def _initialise_config_parameters(self):
        self.dir_file = self.load_config_val(ConfigParams.dir_file)
        self.dir_type = self.load_config_val(ConfigParams.dir_type, "weights")
        self.surf_file = self.load_config_val(ConfigParams.surf_file)
        self.x_ignore = self.load_config_val(ConfigParams.x_ignore)
        self.x_norm = self.load_config_val(ConfigParams.x_norm)
        self.x_cords = self.load_config_val(ConfigParams.x_coord)
        self.y_ignore = self.load_config_val(ConfigParams.y_ignore)
        self.y_norm = self.load_config_val(ConfigParams.y_norm)
        self.y_cords = self.load_config_val(ConfigParams.y_coord)
        self.same_dir = make_bool(self.load_config_val(ConfigParams.same_dir))

    @load_none_on_error
    def load_config_val(self, config_param, default_value=None):
        return self.config_manager.get_config_parameter(config_param, default_value=default_value)

    def check_dir_file_exists(self):
        return self.dir_file and os.path.exists(self.dir_file)

    def get_x_co_ordinates(self):
        return [float(a) for a in self.x_cords.split(':')]

    def get_y_co_ordinates(self):
        if self.y_cords:
            return [float(a) for a in self.y_cords.split(':')]

        return None

    def modify_dir_names_x_params(self):
        dir_file_path = '_' + self.dir_type
        if self.x_ignore:
            dir_file_path += '_xignore=' + str(self.x_ignore)
        if self.x_norm:
            dir_file_path += '_xnorm=' + self.x_norm
        return dir_file_path

    def modify_dir_names_y_params(self):
        dir_file_path = ""
        if self.y_ignore:
            dir_file_path += '_yignore=' + str(self.y_ignore)
        if self.y_norm:
            dir_file_path += '_ynorm=' + self.y_norm
        # y-direction is the same as x-direction
        if self.same_dir:
            dir_file_path += 'same_dir=' + str(self.same_dir)
        return dir_file_path

    def is_direction_setup_needed(self, dir_file):
        if os.path.exists(dir_file):
            f = h5py.File(dir_file, 'r')
            if (self.y_cords and 'ydirection' in f.keys()) or 'xdirection' in f.keys():
                f.close()
                print(f"{dir_file} is already set up")
                return False

        return True

    def create_target_direction(self, meta_model, model1, model2):
        """
            Setup a target direction from one model to the other
            Returns:
              direction: the target direction from net to net2 with the same dimension
                         as weights or states.
        """

        # direction between net2 and net
        net_1, weights_1, state_1 = meta_model.get_model_and_parameters(model1)
        net_2, weights_2, state_2 = meta_model.get_model_and_parameters(model2)

        if self.dir_type and self.dir_type == 'weights':
            direction = get_diff_weights(weights_1, weights_2)
        else:
            direction = get_diff_states(state_1, state_2)

        return direction

    def create_random_direction(self, meta_model, model, ignore='biasbn', norm='filter'):
        """
            Setup a random (normalized) direction with the same dimension as
            the weights or states.

            Returns:
              direction: a random direction with the same dimension as weights or states.
        """

        # random direction
        net, weights, state = meta_model.get_model_and_parameters(model)
        if self.dir_type and self.dir_type == 'states':
            direction = get_random_states(state)
            self.normalize_directions_for_states(direction, state, norm, ignore)
        else:
            direction = get_random_weights(weights)
            self.normalize_directions_for_weights(direction, weights, norm, ignore)

        return direction

    def create_plotting_directions(self, meta_model, dir_file):
        with h5py.File(dir_file, 'w') as f:
            if not self.dir_file:
                print("Setting up the plotting directions...")
                xdirection = self.create_plotting_directions_on_coordinate(meta_model, self.x_ignore, self.x_norm)
                write_list(f, 'xdirection', xdirection)

                if self.y_cords:
                    if self.same_dir:
                        ydirection = xdirection
                    else:
                        ydirection = self.create_plotting_directions_on_coordinate(meta_model, self.y_ignore,
                                                                                   self.y_norm)
                    write_list(f, 'ydirection', ydirection)

    def create_plotting_directions_on_coordinate(self, meta_model, ignore, norm):
        if meta_model.check_model_exist(Models.Model2):
            direction = self.create_target_direction(meta_model, Models.Model1, Models.Model2)
        else:
            direction = self.create_random_direction(meta_model, Models.Model1, ignore, norm)
        return direction

    @staticmethod
    def normalize_direction(direction, weights, norm='filter'):
        """
            Rescale the direction so that it has similar norm as their corresponding
            model in different levels.

            Args:
              direction: a variables of the random direction for one layer
              weights: a variable of the original model for one layer
              norm: normalization method, 'filter' | 'layer' | 'weight'
        """
        if norm == 'filter':
            # Rescale the filters (weights in group) in 'direction' so that each
            # filter has the same norm as its corresponding filter in 'weights'.
            for d, w in zip(direction, weights):
                d.mul_(w.norm() / (d.norm() + 1e-10))
        elif norm == 'layer':
            # Rescale the layer variables in the direction so that each layer has
            # the same norm as the layer variables in weights.
            direction.mul_(weights.norm() / direction.norm())
        elif norm == 'weight':
            # Rescale the entries in the direction so that each entry has the same
            # scale as the corresponding weight.
            direction.mul_(weights)
        elif norm == 'dfilter':
            # Rescale the entries in the direction so that each filter direction
            # has the unit norm.
            for d in direction:
                d.div_(d.norm() + 1e-10)
        elif norm == 'dlayer':
            # Rescale the entries in the direction so that each layer direction has
            # the unit norm.
            direction.div_(direction.norm())

    @staticmethod
    def normalize_directions_for_states(direction, states, norm='filter', ignore='ignore'):
        assert (len(direction) == len(states))
        for d, (k, w) in zip(direction, states.items()):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)  # ignore directions for weights with 1 dimension
                else:
                    d.copy_(w)  # keep directions for weights/bias that are only 1 per node
            else:
                PlotDirections.normalize_direction(d, w, norm)

    @staticmethod
    def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
        """
            The normalization scales the direction entries according to the entries of weights.
        """
        assert (len(direction) == len(weights))
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)  # ignore directions for weights with 1 dimension
                else:
                    d.copy_(w)  # keep directions for weights/bias that are only 1 per node
            else:
                PlotDirections.normalize_direction(d, w, norm)