"""Utility functions for attiicc."""


import os
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import re

from typing import List, Union
from tqdm import tqdm
from PIL import Image
from os.path import join, exists, isdir, isfile, basename, dirname

# --------------------------------------------------------------------------- #
# Internal utility functions

def _get_path_without_ext(file: str) -> str:
    """Get the filename without the extension."""
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