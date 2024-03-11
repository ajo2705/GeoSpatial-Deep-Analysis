import h5py

from loss_landscape import h5_util


def name_direction_file(dir_plotter, meta_models):
    """ Name the direction file that stores the random directions. """

    if dir_plotter.check_dir_file_exists():
        return dir_plotter.dir_file

    dir_file = ""
    dir_file += meta_models.get_x_dir_name_from_model_names()
    dir_file += dir_plotter.modify_dir_names_x_params()

    # name for ydirection
    dir_file += meta_models.get_y_dir_name_from_model_names()
    dir_file += dir_plotter.modify_dir_names_y_params()

    # # index number
    # TODO: implement if needed
    # if args.idx > 0: dir_file += '_idx=' + str(args.idx)

    dir_file += ".h5"
    return dir_file


def setup_direction(dir_file, dir_plotter, meta_model):
    """
    Set up the h5 file to store the directions.
    - xdirection, ydirection: The pertubation direction added to the model.
    The direction is a list of tensors.
    """
    print('-------------------------------------------------------------------')
    print('setup_direction')
    print('-------------------------------------------------------------------')

    # Skip if the direction file already exists
    if not dir_plotter.is_direction_setup_needed(dir_file):
        return

    # Create the plotting directions
    dir_plotter.create_plotting_directions(meta_model, dir_file)
    print("direction file created: %s" % dir_file)


def load_directions(dir_file):
    """ Load direction(s) from the direction file."""

    f = h5py.File(dir_file, 'r')
    if 'ydirection' in f.keys():  # If this is a 2D plot
        xdirection = h5_util.read_list(f, 'xdirection')
        ydirection = h5_util.read_list(f, 'ydirection')
        directions = [xdirection, ydirection]
    else:
        directions = [h5_util.read_list(f, 'xdirection')]

    return directions


def name_surface_file():
    pass


def setup_surface_file():
    pass

