import os
import attiicc as ac
import matplotlib.pyplot as plt
from attiicc.utils import grid_detection

### Load weights and build segmenter
weights = '/home/ecdyer/PROJECTS/nanowell_processing/weights/sam_vit_h_4b8939.pth'
sam = ac.SamSegmenter(weights)

# Load the grid definition
# grid_def = ac.GridDefinition.load('/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/nanowell.npz')

# Load TIF images into a list
all_images = '/home/ecdyer/labshare/WETLAB/WetLab_Shared/NanoExp1b20240209/Raw'
all_d3 = [os.path.join(all_images, d) for d in os.listdir(all_images) if 'd3' in d and os.path.isdir(os.path.join(all_images, d))]

# Want to get plate stacks for each plate (field of view)
# Need images as a list of path strings
# Also want to save each well's image stack

reference_grid_dir = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/visualizations'
# Loop through each plate
for i, field in enumerate(all_d3):
    # Generate a new grid for the field:
    if i == 0:
        grid_def = grid_detection(weights, 
                                  field,  
                                  '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/nanowell.npz')

    # Get field ID
    field_id = field[:4]
    field_time_point_images = [os.path.join(field, f) for f in os.listdir(field) if f.endswith('.TIF')]
    # Build well plates for multiple images,
    # using the same grid detection
    plates = sam.build_plates(field_time_point_images, grid_def)

    # Remove edge wells from first plate
    plates[0].remove_edge_wells()

    # Save the grids for all plates for manual inspection
    # Create a new directory for the grids of each timepoint in the field
    grid_dir = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/visualizations/{field_id}_grids'.format(field_id=field_id)
    os.makedirs(grid_dir, exist_ok=True)
    for j, plate in enumerate(plates):
        save_path = os.path.join(grid_dir, '{field_id}_plate_grid_{j}.png'.format(field_id=field_id, j=j))
        plate.plot(save_path=save_path)
    
    # Stack the plates to ensure that wells are
    # aligned and in the same order
    stack = ac.PlateStack(*plates)

    # Save the stack of each well in the plate (field of view)
    stack.save_all_wells('/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/stacks/{field_id}_stack'.format(field_id))
