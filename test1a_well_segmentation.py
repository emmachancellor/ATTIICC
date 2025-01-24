import os
import attiicc as ac
from attiicc import SamSegmenter
import attiicc.utils as acu
import matplotlib.pyplot as plt
import requests

channels = ['d0', 'd2', 'd3']
for channel in channels:    
    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data=f"Beginning well segmentation for {channel}.".encode(encoding='utf-8'))
    print(f"Beginning well segmentation for {channel}...")

    ### Load weights and build segmenter
    weights = '/home/ecdyer/PROJECTS/nanowell_processing/weights/sam_vit_h_4b8939.pth'
    sam = ac.SamSegmenter(weights)

    # Specify the save paths
    well_save_path = f'/home/ecdyer/labshare/WETLAB/WetLab_Shared/test_1a/tif_well_segmentation_{channel}/stacks'
    grid_def_path = f'/home/ecdyer/labshare/WETLAB/WetLab_Shared/NanoExp1a20240110/well_segmentation/reference_grids'
    grid_vis_path = f'/home/ecdyer/labshare/WETLAB/WetLab_Shared/test_1a/tif_well_segmentation_{channel}/visualizations'

    # Get the field directories
    data_dir = '/home/ecdyer/labshare/WETLAB/WetLab_Shared/test_1a/data'

    # Select the correct field directories (d0, d1, d2, or d3)
    field_dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if channel in f and os.path.isdir(os.path.join(data_dir, f))]
    print('Field directories:', field_dirs)

    for field_dir in field_dirs:
        acu.segment_field(field_dir=field_dir, 
                    sam=sam, 
                    grid_def_path=grid_def_path, 
                    well_save_path=well_save_path,
                    well_file_type='tif',
                    grid_vis_path=grid_vis_path,
                    use_existing_grids=True)

    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data=f"Well segmentation complete for {channel}.".encode(encoding='utf-8'))
    print(f"Well segmentation complete for {channel}.")