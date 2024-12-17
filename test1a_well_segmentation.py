import os
import attiicc as ac
from attiicc import SamSegmenter
import attiicc.utils as acu
import matplotlib.pyplot as plt
import requests

requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Beginning well segmentation.".encode(encoding='utf-8'))
print("Beginning well segmentation...")

### Load weights and build segmenter
weights = '/home/ecdyer/PROJECTS/nanowell_processing/weights/sam_vit_h_4b8939.pth'
sam = ac.SamSegmenter(weights)

well_save_path = '/home/ecdyer/labshare/WETLAB/WetLab_Shared/test_1a//well_segmentation_d0/stacks'
grid_def_path = '/home/ecdyer/labshare/WETLAB/WetLab_Shared/NanoExp1a20240110/well_segmentation/reference_grids'
grid_vis_path = '/home/ecdyer/labshare/WETLAB/WetLab_Shared/test_1a/well_segmentation_d0/visualizations'

# Get the field directories
data_dir = '/home/ecdyer/labshare/WETLAB/WetLab_Shared/test_1a/data'
field_dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'd0' in f and os.path.isdir(os.path.join(data_dir, f))]
print('Field directories:', field_dirs)

for field_dir in field_dirs:
    acu.segment_field(field_dir=field_dir, 
                  sam=sam, 
                  grid_def_path=grid_def_path, 
                  well_save_path=well_save_path, 
                  grid_vis_path=grid_vis_path,
                  use_existing_grids=True)

requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Well segmentation complete.".encode(encoding='utf-8'))
print("Well segmentation complete.")