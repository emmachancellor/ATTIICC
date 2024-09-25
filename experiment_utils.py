'''Utility functions for attiicc.Experiment.'''


import os
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import attiicc as ac
import re
from PIL import Image

def convert_tif_to_png(tif_path: str, 
                       png_path: str = None, 
                       single_image: bool = False) -> None:
    '''
    Convert a .tif image to a .png image.
    Inputs:
        tif_path: (str) The path to a directory of .tif images.
        png_path: (str) The path to a directory of .png images. Default is None.
            If None, a new directory will be created at the same level as the
            tif_path directory, with '_png' appended to the name.
        single_image: (bool) If True, then tif_path is a path to a single .tif
    Outputs:
        png_path: (str) The path to a directory of .png images.
    '''
    if single_image is False:
        files = os.listdir(tif_path)
        total = len(files)
        if png_path is None:
            png_path = tif_path + '_png'
        if not os.path.exists(png_path):
            os.makedirs(png_path)
    else:
        files = [tif_path]
        if png_path is None:
            png_path = tif_path.rstrip(".TIF") + '_png'
    for i, f in enumerate(files):
        if '.TIF' in f and '._' not in f:
            print(f'({i}/{total})  Converting image {f} to .png')
            name = f'{tif_path}/{f}'
            im = np.array(Image.open(name))
            im = Image.fromarray(im / np.amax(im) * 255)
            im = im.convert("L")

            # Rename file
            #file_name = str(f.replace(' ', '')).rstrip(".TIF")
            file_name = str(f).rstrip(".TIF")
            print(file_name)
            new_name = f'{png_path}/{file_name}'
            print(new_name)
            im.save(new_name + '.png', 'PNG')
    return png_path

def find_files(directory: str, 
                pattern1: str, 
                pattern2: str, 
                pattern3: str):
    '''
    Find files in a directory that match two patterns.
    Inputs:
        directory: (str) The directory to search.
        pattern1: (str) The first pattern to match.
        pattern2: (str) The second pattern to match.
        pattern3: (str) The third pattern to match.
    Outputs:
        filename: (str) The path to a file that matches both patterns.
    '''
    matching_fies = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, f'*{pattern1}*') and \
                fnmatch.fnmatch(basename, f'*{pattern2}*') and \
                fnmatch.fnmatch(basename, f'*{pattern3}*'):
                filename = os.path.join(root, basename)
                print("appending:", filename)
                matching_fies.append(filename)
    return matching_fies

def generate_comparison_plot(image_1: str, 
                             image_2: str,
                             time_point_1: str = None,
                             time_point_2: str = None,
                             field_of_view: str = None,
                             figsize: tuple = (10, 5),
                             save_path: str = None) -> None:
    '''
    Create a plot of two images side by side. This is meant to be 
    used to compare images from different time points that have
    discordant ROI coordinates.

    Inputs:
        image_1: (str) The path to the first image.
        image_2: (str) The path to the second image.
        time_point: (int) The time point of the first image.
        field_of_view: (str) The field of view of the first image.
        figsize: (tuple) The size of the figure.
        save_path: (str) The path to save the plot. If None, the plot
            will not be saved.
    Outputs:
        None
    '''
    #Load images
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
    # Hide the axis ticks and labels for better presentation
    for ax in axes:
        ax.axis('off')
    # Adjust the layout to prevent overlap of titles
    plt.tight_layout()
    # Show the plot
    plt.show()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        plt.savefig(save_path + f'/comparison_{field_of_view}_{time_point_1}.png')
    return

def sort_paths(paths):
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