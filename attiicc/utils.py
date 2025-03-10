"""Utility functions for attiicc."""


import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import re
import attiicc as ac

from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
from os.path import join, exists, isdir, basename, dirname

# --------------------------------------------------------------------------- #
# Internal utility functions

def _get_path_without_ext(file: str) -> str:
    """Get the full file path without the extension."""
    return join(dirname(file), '.'.join(basename(file).split('.')[:-1]))


def _get_filename_without_ext(file: str) -> str:
    """Get the filename without the extension."""
    return '.'.join(basename(file).split('.')[:-1])


# --------------------------------------------------------------------------- #
# User-facing functions

def is_tif(file: str) -> bool:
    """Check if a file is a .tif or .tiff file.

    Args:
        file: (str) The name of the file.

    Returns:
        (bool) True if the file is a .tif or .tiff file, False otherwise.

    """
    return file.lower().endswith('.tif') or file.lower().endswith('.tiff')


def load_tif(file: str) -> Image:
    """Load a .tif or .tiff file.

    Args:
        file: (str) The name of the file.

    Returns:
        (Image) The image.

    """
    # Load the image
    img = np.array(Image.open(file))

    # Normalize the image
    img = Image.fromarray(img / np.amax(img) * 255)

    # Convert the image to grayscale
    img = img.convert("L")

    return img


def convert_tif_to_png(tif_path: str, png_path: str = None) -> str:
    """Convert a .tif image to a .png image.

    Args:
        tif_path: (str) The path to a directory of .tif images.
        png_path: (str) The path to a directory of .png images. Default is None.
            If None, a new directory will be created at the same level as the
            tif_path directory, with '_png' appended to the name.

    Returns:
        png_path: (str) The path to a directory of .png images.

    """
    # If the input is a directory, convert all .tif images in the directory
    if isdir(tif_path):
        files = [join(tif_path, f) for f in os.listdir(tif_path)]
        # Store files in a folder ending in "_png"
        if png_path is None:
            png_path = tif_path + '_png'
        if not exists(png_path):
            os.makedirs(png_path)
    else:
        files = [tif_path]
        if png_path is None:
            # Store files in a folder ending in "_png",
            # excluding the file extension (TIF, tiff, etc)
            png_path = _get_path_without_ext(tif_path) + '_png'

    # Only use valid .tif files
    files = [f for f in files
             if is_tif(f) and not basename(f).startswith('.')]

    # Convert .tif images to .png images
    for i, filename in tqdm(enumerate(files), desc='Converting images...'):
        # Load the TIFF image
        img = load_tif(filename)

        # Save to PNG
        dest = join(png_path, _get_filename_without_ext(filename) + '.png')
        img.save(dest, 'PNG')

    return png_path


def generate_comparison_plot(
    image_1: str,
    image_2: str,
    time_point_1: str = None,
    time_point_2: str = None,
    field_of_view: str = None,
    figsize: tuple = (10, 5),
    save_path: str = None
) -> None:
    """
    Create a plot of two images side by side. This is meant to be
    used to compare images from different time points that have
    discordant ROI coordinates.

    Args:
        image_1: (str) The path to the first image.
        image_2: (str) The path to the second image.
        time_point: (int) The time point of the first image.
        field_of_view: (str) The field of view of the first image.
        figsize: (tuple) The size of the figure.
        save_path: (str) The path to save the plot. If None, the plot
            will not be saved.

    Returns:
        None

    """
    # Load images
    img1 = mpimg.imread(image_1)
    img2 = mpimg.imread(image_2)

    # Create a figure and set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjust the figsize as needed

    # Display the first image in the first subplot
    axes[0].imshow(img1)
    axes[0].set_title(f'Field {field_of_view} at Time Point {time_point_1}')  # Add a title to the first subplot

    # Display the second image in the second subplot
    axes[1].imshow(img2)
    axes[1].set_title(f'Field {field_of_view} at Time Point {time_point_2}')  # Add a title to the second subplot

    # Hide the axis ticks and labels for better presentation,
    # and adjust the layout to prevent overlap of titles
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

    # Show the plot.
    plt.show()

    # Save the plot.
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
        dest = save_path + f'/comparison_{field_of_view}_{time_point_1}.png'
        plt.savefig(dest)
        print("Figure saved to {}".format(dest))


def sort_paths(paths: List[str]):
    """
    Sorts a list of file paths based on the numeric value found after the character 'p' in the filenames.
    Args:
        paths (list of str): A list of file paths to be sorted.
    Returns:
        list of str: The sorted list of file paths.
    Example:
        paths = [
            "/path/to/file_p10.txt",
            "/path/to/file_p2.txt",
            "/path/to/file_p1.txt"
        ]
        sorted_paths = sort_paths(paths)
        # sorted_paths will be:
        # [
        #     "/path/to/file_p1.txt",
        #     "/path/to/file_p2.txt",
        #     "/path/to/file_p10.txt"
        # ]
    """

    def extract_number(path):
        # Extract the filename from the path
        filename = path.split('/')[-1]

        # Find the number after 'p' in the filename
        match = re.search(r'p(\d+)', filename)

        # Return the number as an integer, or 0 if not found
        return int(match.group(1)) if match else 0

    # Sort the paths using the extract_number function as key
    return sorted(paths, key=extract_number)

def grid_detection(weights, reference_img, grid_def_path, remove_well_threshold=0.9, save_path=None):
    """
    Detects the grid of a well plate in a reference image, saves the plot of the detected grid,
    and saves the grid definition to a specified path.
    Args:
        weights (str): Path to the weights file for the segmenter.
        reference_img (str): Path to the reference image file.
        save_path (str): Path where the plot of the detected grid will be saved.
        grid_def_path (str): Path where the grid definition will be saved.
    Returns:
        GridDefinition: The grid definition object containing the details of the detected grid.
    """

    ### Load weights and build segmenter
    sam = ac.SamSegmenter(weights)

    ### Detect the well plate grid
    # Segment the image
    segmentation = sam.segment(reference_img)

    # Find the well plate regions.
    # Each region will have its own shape and centroid
    # as detected by SAM
    rough_plate = segmentation.find_wells()

    # Build the well plate grid
    # This auto-detects the average well plate shape
    # and creates a grid of regions 
    # using this average shape
    plate = rough_plate.build_grid()

    # Remove edge wells
    plate.remove_edge_wells(threshold=remove_well_threshold)

    # Save the plot
    if save_path is not None:
        plate.plot(save_path=save_path)

    # Save the grid definition
    grid_def = plate.grid_definition
    print(grid_def_path)
    grid_def.save(grid_def_path)
    return grid_def

def segment_field(field_dir: str,
                  sam,
                  field_ref_grid_key: str = 'p00',
                  grid_def_path: str = None,
                  well_save_path: str = None,
                  well_file_type: str = 'png',
                  grid_vis_path: str = None,
                  use_existing_grids: bool = False,
                  use_og_img: bool = False,
                  area_range: Tuple[int, int] = (10000,20000),
                  filter_distance: int = 10) -> None:
    """
    Segment a field of images.
    """
    from .sam import SamSegmenter
    # Get images in the field
    field_part = os.path.basename(field_dir)
    field_id = field_part.split('f')[1].split('d')[0]
    print('Field ID:', field_id)
    img_list = [os.path.join(field_dir, f) for f in os.listdir(field_dir) if f.endswith('.TIF')]
    
    if use_existing_grids is False:
        # Path to save reference grid for the field
        # Create grid definition directory if it doesn't exist
        if not os.path.exists(grid_def_path):
            os.makedirs(grid_def_path)
        file_grid_def = os.path.join(grid_def_path, f'{field_id}_nanowell.npz')
        print('Grid definition path:', file_grid_def)
        # if os.path.exists(file_grid_def):
        #     print('Grid definition already exists, skipping to next field...')
            
        print('Images in field:', img_list)
        # Create a reference grid for the field
        reference_grid_img = next(img for img in img_list if field_ref_grid_key in img)
        print('Reference grid image:', reference_grid_img)
        segmentation = sam.segment(reference_grid_img)
        rough_plate = segmentation.find_wells(area_range=area_range, filter_distance=filter_distance)
        plate = rough_plate.build_grid()
        plate.remove_edge_wells()
        grid_def = plate.grid_definition
        print('Saving grid definition to:', file_grid_def)
        grid_def.save(file_grid_def)
    else:
        reference_grids = os.listdir(grid_def_path)
        ref_grid = next(grid for grid in reference_grids if field_id in grid)
        grid_def = ac.GridDefinition.load(os.path.join(grid_def_path, ref_grid))

    plates = sam.build_plates(img_list, grid_def, use_og_img, area_range, filter_distance)
    plates[0].remove_edge_wells()
    for j, p in enumerate(plates):
        grid_dir = os.path.join(grid_vis_path, f'{field_id}_grids')
        if not os.path.exists(grid_dir):
            os.makedirs(grid_dir)
        save_path = os.path.join(grid_dir, '{field_id}_plate_grid_{j}.png'.format(field_id=field_id, j=j))
        p.plot(save_path=save_path)   

    if well_save_path is not None:
        # Save individual wells
        stack = ac.PlateStack(*plates)
        # Create save directory if it doesn't exist
        if not os.path.exists(os.path.join(well_save_path, f'{field_id}_stack')):
            os.makedirs(os.path.join(well_save_path, f'{field_id}_stack'))
        stack.save_all_wells(os.path.join(well_save_path, f'{field_id}_stack'),
                             well_file_type=well_file_type)
    return