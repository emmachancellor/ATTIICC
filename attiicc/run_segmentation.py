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
#all_images = '/home/ecdyer/labshare/WETLAB/WetLab_Shared/NanoExp1b20240209/Raw'
all_images = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/smaller_test_data'
all_d3 = [os.path.join(all_images, d) for d in os.listdir(all_images) if 'd3' in d and os.path.isdir(os.path.join(all_images, d))]

# Want to get plate stacks for each plate (field of view)
# Need images as a list of path strings
# Also want to save each well's image stack

reference_grid_dir = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/reference_grids'
# Loop through each plate
for i, field in enumerate(all_d3):
    print(f'Processing field {field} ({i+1}/{len(all_d3)})')

    # Get field ID
    field_id = field[-5:]  # Extract the relevant part of the field path
    print(field_id)
    field_time_point_images = [os.path.join(field, f) for f in os.listdir(field) if f.endswith('.TIF')]
    
    # Generate a new grid for the field:
    if i == 0:
        referece_grid_path = os.path.join(reference_grid_dir, f'{field_id}_nanowell.npz')
        grid_def = grid_detection(weights, 
                                  field_time_point_images[0],  
                                  referece_grid_path,
                                  remove_well_threshold=0.9)
    
    # Build well plates for multiple images,
    # using the same grid detection
    plates = sam.build_plates(field_time_point_images, grid_def)

    # Save the grids for all plates for manual inspection
    # Create a new directory for the grids of each timepoint in the field
    grid_dir = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/visualizations/{field_id}_grids'.format(field_id=field_id)
    os.makedirs(grid_dir, exist_ok=True)
    for j, plate in enumerate(plates):
        print(f'Processing plate {j+1}/{len(plates)}')
        if j > 10: 
            break
        # Remove edge wells from all plates (? not sure if this is necessary, trying to debug misalignment)
        # plates[j].remove_edge_wells()
        save_path = os.path.join(grid_dir, '{field_id}_plate_grid_{j}.png'.format(field_id=field_id, j=j))
        plate.plot(save_path=save_path)
    
    # Stack the plates to ensure that wells are
    # aligned and in the same order
    stack = ac.PlateStack(*plates)

    # Save the stack of each well in the plate (field of view)
    stack.save_all_wells(f'/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/stacks/{field_id}_stack')
