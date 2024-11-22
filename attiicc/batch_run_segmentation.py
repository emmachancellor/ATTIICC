import os
import attiicc as ac
import matplotlib.pyplot as plt
from attiicc.utils import grid_detection

def segment_field(field_dir: str,
                  sam: ac.SamSegmenter,
                  grid_def_path: str = None,
                  well_save_path: str = None) -> None:
    """
    Segment a field of images.
    """
    # Get images in the field
    field_part = os.path.basename(field_dir)
    field_id = field_part.split('f')[1].split('d')[0]
    print('Field ID:', field_id)

    # Path to save reference grid for the field
    grid_def_path = os.path.join(grid_def_path, f'{field_id}_nanowell.npz')
    print('Grid definition path:', grid_def_path)
    img_list = [os.path.join(field_dir, f) for f in os.listdir(field_dir) if f.endswith('.TIF')]
    print('Images in field:', img_list)

    # Create a reference grid for the field
    reference_grid_img = next(img for img in img_list if 'p00' in img)
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

    if well_save_path is not None:
        # Save individual wells
        stack = ac.PlateStack(*plates)
        # Create save directory if it doesn't exist
        if not os.path.exists(os.path.join(well_save_path, f'{field_id}_stack')):
            os.makedirs(os.path.join(well_save_path, f'{field_id}_stack'))
        stack.save_all_wells(os.path.join(well_save_path, f'{field_id}_stack'))
    return


### Load weights and build segmenter
weights = '/home/ecdyer/PROJECTS/nanowell_processing/weights/sam_vit_h_4b8939.pth'
sam = ac.SamSegmenter(weights)

well_save_path = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/stacks'
grid_def_path = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/reference_grids'

# Get the field directories
data_dir = '/home/ecdyer/PROJECTS/nanowell_processing/exp1b_PDL1/smaller_test_data'
field_dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'd3' in f and os.path.isdir(os.path.join(data_dir, f))]
print('Field directories:', field_dirs)

for field_dir in field_dirs:
    segment_field(field_dir, sam, grid_def_path, well_save_path)
