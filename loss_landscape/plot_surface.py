import torch
import h5py
import numpy as np
import os
from torch.utils.data import DataLoader

from load_data import load_train_linear_data
from loss_landscape import net_plotter
from loss_landscape import projections as proj
from loss_landscape.plot_directions import PlotDirections
from loss_landscape.trained_models import TrainedModels
from loss_landscape.plot_2D import plot_2d_contour
from libraries.config_manager import ConfigManager, ConfigError
from libraries.constants import ConfigParams


def get_plotting_resolution(config_manager: ConfigManager):
    x_cords = config_manager.get_config_parameter(ConfigParams.x_coord)
    y_cords = config_manager.get_config_parameter(ConfigParams.y_coord)

    try:
        x_min, x_max, x_num = [float(a) for a in x_cords.split(':')]
        y_min, y_max, y_num = [float(a) for a in y_cords.split(':')]

        assert y_min and y_max and y_num , "You specified some arguments for the y axis, but not all"

    except Exception as e:
        print(e.__repr__())
        return ConfigError(f"Error in configuration of {ConfigParams.direction} in {config_manager.config_file}")


def name_surface_file(dir_plotter, dir_file):
    # skip if surf_file is specified in args
    if dir_plotter.surf_file:
        return dir_plotter.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (*dir_plotter.get_x_co_ordinates(), )
    y_coords = dir_plotter.get_y_co_ordinates()

    if y_coords:
        surf_file += 'x[%s,%s,%d]' % (*y_coords, )
    return surf_file + ".h5"


def is_surface_setup_needed(surf_file, y_cords):
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (y_cords and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print(f"{surf_file} is already set up")
            return False

    return True


def setup_surface_file(surf_file, dir_file, dir_plotter):
    # skip if direction file is needed
    y_coords = dir_plotter.get_y_co_ordinates()
    if not is_surface_setup_needed(surf_file, y_coords):
        return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    x_min, x_max, x_num = (*dir_plotter.get_x_co_ordinates(), )
    x_coordinates = np.linspace(x_min, x_max, num=int(x_num))
    f['xcoordinates'] = x_coordinates

    if y_coords:
        y_min, y_max, y_num = (*y_coords, )
        y_coordinates = np.linspace(y_min, y_max, num =int( y_num))
        f['ycoordinates'] = y_coordinates
    f.close()

    return surf_file


def load_directions(dir_file):
    directions = net_plotter.load_directions(dir_file)
    similarity = None
    if len(directions) == 2:
        similarity = proj.cal_angle(proj.nplist_to_tensor(directions[0]), proj.nplist_to_tensor(directions[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    return directions, similarity


def compute(surf_file, plot_dir, meta_model, directions):
    """
     Calculate the loss values and accuracies of modified models
    """
    f = h5py.File(surf_file, 'r+')

    # calculate train loss and train acc
    dataloader = DataLoader(dataset=load_train_linear_data(), batch_size=128, shuffle=True)
    meta_model.load_surface_file_loss(f, surf_file, plot_dir.dir_type, directions, dataloader, "train_acc")


def plot_figures(surf_file):
    plot_2d_contour(surf_file, "train_loss")


def plot_surface():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_manager = ConfigManager()
    plot_dir = PlotDirections(config_manager)
    meta_model = TrainedModels(config_manager, device)

    dir_file = net_plotter.name_direction_file(plot_dir, meta_model)
    net_plotter.setup_direction(dir_file, plot_dir, meta_model)
    surf_file = name_surface_file(plot_dir, dir_file)
    setup_surface_file(surf_file, dir_file, plot_dir)
    directions, similarity = load_directions(dir_file)

    compute(surf_file, plot_dir, meta_model, directions)
    plot_figures(surf_file)
