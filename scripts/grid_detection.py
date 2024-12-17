import attiicc as ac
import matplotlib.pyplot as plt

def grid_detection(weights, reference_img, save_path, grid_def_path):
    """
    Detects the grid of a well plate in a reference image, saves the plot of the detected grid,
    and saves the grid definition to a specified path.
    Args:
        weights (str): Path to the weights file for the segmenter.
        reference_img (str): Path to the reference image file.
        save_path (str): Path where the plot of the detected grid will be saved.
        grid_def_path (str): Path where the grid definition will be saved.
    Returns:
        GridDefinition: The grid definition object containing the details of the detected grid.
    """

    ### Load weights and build segmenter
    sam = ac.SamSegmenter(weights)

    ### Detect the well plate grid
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
    plate.plot(save_path=save_path)

    # Save the grid definition
    grid_def = plate.grid_definition
    grid_def.save(grid_def_path)
    return grid_def