'''Utility functions for attiicc.Experiment.'''


import os
import numpy as np
import attiicc as ac
from PIL import Image

def convert_tif_to_png(tif_path: str, png_path: str = None) -> None:
    '''
    Convert a .tif image to a .png image.
    Inputs:
        tif_path: (str) The path to a directory of .tif images.
        png_path: (str) The path to a directory of .png images. Default is None.
            If None, a new directory will be created at the same level as the
            tif_path directory, with '_png' appended to the name.
    Outputs:
        None
    '''
    files = os.listdir(tif_path)
    total = len(files)
    file_names = []
    if png_path is None:
        png_path = tif_path + '_png'
    if not os.path.exists(png_path):
        os.makedirs(png_path)
    for i, f in enumerate(files):
        if '.TIF' in f:
            print(f'({i}/{total})  Converting image {f} to .png')
            name = f'{tif_path}/f'
            im = np.array(Image.open(name))
            im = Image.fromarray(im / np.amax(im) * 255)
            im = im.convert("L")

            # Rename file
            file_name = str(f).rstrip(".TIF")
            new_name = f'{png_path}/{file_name}'
            print(new_name)
            im.save(new_name + '.png', 'PNG')
    return None
