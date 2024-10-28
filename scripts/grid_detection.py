import attiicc as ac
import matplotlib.pyplot as plt

### Load weights and build segmenter
weights = '/home/ecdyer/PROJECTS/nanowell_processing/weights/sam_vit_h_4b8939.pth'
sam = ac.SamSegmenter(weights)

### Detect the well plate grid
# Use a reference image to detect the well plate grid
reference_img = '/home/ecdyer/labshare/WETLAB/WetLab_Shared/NanoExp1b20240209/Raw/f02d3/scan_Top%20Slide_R_p00_0_A01f02d3.TIF'

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
plate.remove_edge_wells()

# Save the plot
plate.plot(save_path='/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/visualizations/plate_grid_Exp1b_PDL1.png')

# Save the grid definition
grid_def = plate.grid_definition
grid_def.save('/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/nanowell.npz')