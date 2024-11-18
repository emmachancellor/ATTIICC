import os
import attiicc as ac
import matplotlib.pyplot as plt
from attiicc.utils import grid_detection

### Load weights and build segmenter
weights = '/home/ecdyer/PROJECTS/nanowell_processing/weights/sam_vit_h_4b8939.pth'
sam = ac.SamSegmenter(weights)

# Get images
data_dir = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/smaller_test_data'

# Get the field directories
field_dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'd3' in f and os.path.isdir(os.path.join(data_dir, f))]
print('Field directories:', field_dirs)
# Loop through each field
for i, field_dir in enumerate(field_dirs):
    # Get images in the field
    field_part = os.path.basename(field_dir)
    field_id = field_part.split('f')[1].split('d')[0]
    print('Field ID:', field_id)

    # Path to save reference grid for the field
    grid_def_path = os.path.join('/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/reference_grids', f'{field_id}_nanowell.npz')
    print('Grid definition path:', grid_def_path)
    img_list = [os.path.join(field_dir, f) for f in os.listdir(field_dir) if f.endswith('.TIF')]
    print('Images in field:', img_list)

    # Create a reference grid for the field
    reference_grid_img = img_list[0]
    print('Reference grid image:', reference_grid_img)
    segmentation = sam.segment(reference_grid_img)
    rough_plate = segmentation.find_wells()
    plate = rough_plate.build_grid()
    plate.remove_edge_wells()
    grid_def = plate.grid_definition
    print(grid_def_path)
    grid_def.save(grid_def_path)

    # Build plates for the field
    plates = sam.build_plates(img_list, grid_def)
    plates[0].remove_edge_wells()
    for j, p in enumerate(plates):
        grid_dir = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/visualizations/{field_id}_grids'.format(field_id=field_id)
        if not os.path.exists(grid_dir):
            os.makedirs(grid_dir)
        save_path = os.path.join(grid_dir, '{field_id}_plate_grid_{j}.png'.format(field_id=field_id, j=j))
        p.plot(save_path=save_path)

    # Save individual wells
    stack = ac.PlateStack(*plates)
    stack.save_all_wells(f'/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/stacks/{field_id}_stack')
    


