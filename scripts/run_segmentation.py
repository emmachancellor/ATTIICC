import os
import attiicc as ac
import matplotlib.pyplot as plt

### Load weights and build segmenter
weights = '/home/ecdyer/PROJECTS/nanowell_processing/weights/sam_vit_h_4b8939.pth'
sam = ac.SamSegmenter(weights)

# Load the grid definition
grid_def = ac.GridDefinition.load('/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/nanowell.npz')

# Load TIF images into a list
all_images = '/home/ecdyer/labshare/WETLAB/WetLab_Shared/NanoExp1b20240209/Raw'
all_d3 = [os.path.join(all_images, d) for d in os.listdir(all_images) if 'd3' in d and os.path.isdir(os.path.join(all_images, d))]

# Want to get plate stacks for each plate (field of view)
# Need images as a list of path strings
# Also want to save each well's image stack

# Loop through each plate
for plate in all_d3:
    # Get field ID
    field_id = plate[:4]
    plate_time_point_images = [os.path.join(plate, f) for f in os.listdir(plate) if f.endswith('.TIF')]
    # Build well plates for multiple images,
    # using the same grid detection
    plates = sam.build_plates(plate_time_point_images, grid_def)

    # Remove edge wells from first plate
    plates[0].remove_edge_wells()
    
    # Stack the plates to ensure that wells are
    # aligned and in the same order
    stack = ac.PlateStack(*plates)

    # Save the stack of each well in the plate (field of view)
    stack.save('/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/stacks/{field_id}_stack.npz'.format(field_id))
