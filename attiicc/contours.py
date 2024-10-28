import cv2
import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .segmentation import GridDefinition

# -----------------------------------------------------------------------------
# Discover contours

def align_contours(contours: List[np.ndarray]) -> List[np.ndarray]:
    """Align contours to the origin by translating them.

    Args:
        contours: A list of contours, where each contour is an array of points.

    Returns:
        A list of aligned contours.

    """
    aligned_contours = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Translate contour to origin
        aligned_contour = contour - [cX, cY]
        aligned_contours.append(aligned_contour)

    return aligned_contours


def resample_contour(contour: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Resample a contour to have a fixed number of points.

    Args:
        contour (np.ndarray): A contour as an array of points.
        n_points (int): The number of points to resample the contour to.

    Returns:
        np.ndarray: The resampled contour.

    """
    contour = np.array(contour).reshape(-1, 2)

    # Calculate the cumulative length along the contour
    distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Create new points along the contour at regular intervals
    total_length = cumulative_distances[-1]
    regular_intervals = np.linspace(0, total_length, n_points)

    resampled_contour = []
    for interval in regular_intervals:
        index = np.searchsorted(cumulative_distances, interval)
        if index == len(contour):
            resampled_contour.append(contour[-1])
        else:
            p1, p2 = contour[index-1], contour[index]
            d1, d2 = cumulative_distances[index-1], cumulative_distances[index]
            t = (interval - d1) / (d2 - d1) if d2 > d1 else 0
            interpolated_point = p1 + t * (p2 - p1)
            resampled_contour.append(interpolated_point)

    return np.array(resampled_contour, dtype=np.float32)


def calculate_average_contour(
    contours: List[np.ndarray],
    n_points: int = 100,
    method: str = 'median'
) -> np.ndarray:
    """Calculate the average contour from a list of contours.

    Args:
        contours (List[np.ndarray]): A list of contours, where each contour is an array of points.
        n_points (int): The number of points to resample each contour to.
        method (str): The method to use for calculating the average contour. Can be 'mean' or 'median'.

    Returns:
        np.ndarray: The average contour

    """
    # Align contours
    aligned_contours = align_contours(contours)

    # Resample contours
    resampled_contours = [resample_contour(contour, n_points) for contour in aligned_contours]

    # Calculate average points
    reduce_fn = np.mean if method == 'mean' else np.median
    average_contour = reduce_fn(resampled_contours, axis=0)

    # Convert back to integer coordinates
    average_contour = np.round(average_contour).astype(int)

    return average_contour

# -----------------------------------------------------------------------------
# Rotation and alignment

def rotate_coordinates(
    coordinates: np.ndarray,
    angle: float,
    anchor: np.ndarray = None,
    buffer: int = 0
) -> np.ndarray:
    """Rotate a set of coordinates around the origin.

    Args:
        coordinates (np.ndarray): An array of coordinates to rotate.
        angle (float): The angle of rotation in degrees.
        anchor (np.ndarray): The point to rotate the coordinates around.
        buffer (int): The buffer to add around the grid.

    Returns:
        np.ndarray: The rotated coordinates.

    """
    if isinstance(buffer, int):
        buffer = np.array([buffer, buffer])

    # Calculate grid size
    M = np.max(coordinates[:, 0]) + buffer[0]
    N = np.max(coordinates[:, 1]) + buffer[0]
    grid_size = (M, N)

    # Calculate the center of the grid
    if anchor is None:
        anchor = np.array([grid_size[0] // 2, grid_size[1] // 2])

    # Convert angle from degrees to radians
    theta = np.radians(angle)

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Translate coordinates to the center, apply rotation, then translate back
    translated_coords = coordinates - anchor
    rotated_coords = np.dot(translated_coords, rotation_matrix.T)
    final_coords = rotated_coords + anchor

    # Round the coordinates to nearest integer (since grid indices must be integers)
    return np.round(final_coords).astype(int)


def create_rotated_grid(
    coordinates: np.ndarray,
    angle: float,
    radius: int = 3
) -> np.ndarray:
    """Create a grid with squares around the rotated coordinates.

    Args:
        coordinates (np.ndarray): An array of coordinates to rotate.
        angle (float): The angle of rotation in degrees.
        radius (int): The radius of the square around each coordinate.

    Returns:
        np.ndarray: The rotated grid

    """
    # Calculate grid size
    M = np.max(coordinates[:, 0]) + 50
    N = np.max(coordinates[:, 1]) + 50
    grid_size = (M, N)

    # Rotate the coordinates around the center
    rotated_coordinates = rotate_coordinates(coordinates, angle, buffer=50)

    # Initialize the grid
    grid = np.zeros(grid_size, dtype=bool)

    # Place squares around the rotated coordinates
    for x, y in rotated_coordinates:

        # Ensure x and y are within the grid bounds before trying to create the square
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:

            # Define the square region limits, making sure we stay within the grid bounds
            x_min = max(0, x - radius)
            x_max = min(grid_size[0], x + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(grid_size[1], y + radius + 1)

            # Set the region within the square to True
            grid[x_min:x_max, y_min:y_max] = True

    return grid


def find_angle(
    coordinates: np.ndarray,
    angle_min: float = -10,
    angle_max: float = 10,
    angle_step: float = 0.1,
    axis: int = 0
) -> float:
    """Find the angle which minimizes the sum of the maximum values along an axis.

    Args:
        coordinates (np.ndarray): An array of coordinates to rotate.
        angle_min (float): The minimum angle to search.
        angle_max (float): The maximum angle to search.
        angle_step (float): The step size for the search.
        axis (int): The axis to calculate the maximum values along.

    Returns:
        float: The angle which minimizes the sum of the maximum values along the axis

    """
    min_sum = np.inf
    best_angle = 0
    for angle in np.arange(angle_min, angle_max, angle_step):
        grid = create_rotated_grid(coordinates, angle)
        axis0_max = grid.max(axis=axis).sum()
        if axis0_max < min_sum:
            min_sum = axis0_max
            best_angle = angle
    return best_angle


def count_rows(grid: np.ndarray, min_length: int = 5) -> int:
    """Count the number of rows in a grid.

    Args:
        grid (np.ndarray): A binary grid.
        min_length (int): The minimum length of a row.

    Returns:
        int: The number of rows in the grid.

    """
    # Find the maximum value along the 0-axis (x-axis)
    amax = grid.max(axis=0)

    # Identify where the True values are
    padded_arr = np.pad(amax.astype(int), (1, 1), 'constant', constant_values=(0, 0))
    diff = np.diff(padded_arr)

    # Find the start and end indices of each run of True values
    start_indices = np.where(diff == 1)[0]
    end_indices = np.where(diff == -1)[0]

    # Calculate the length of each run
    lengths = end_indices - start_indices

    # Count the number of runs that are at least of the desired length
    count = np.sum(lengths >= min_length)

    return count


def get_median_x_distance(
    coords: np.ndarray,
    grid: np.ndarray,
    n_rows: Optional[int] = None
) -> int:
    """Calculate the median distance between centroids along the x-axis.

    Args:
        coords (np.ndarray): An array of coordinates.
        grid (np.ndarray): A binary grid.
        n_rows (int, optional): The number of rows in the grid.

    Returns:
        int: The median x-distance between centroids.

    """
    if n_rows is None:
        n_rows = count_rows(grid)

    row_width = grid.shape[1] // n_rows

    x_distances = []
    for row in range(n_rows):
        # Define the y-limits for the current row
        y_min = row * row_width
        y_max = (row + 1) * row_width if row < n_rows - 1 else grid.shape[1]

        # Get the centroids within the y-limits
        row_centroids = coords[(coords[:, 1] >= y_min) & (coords[:, 1] < y_max)]

        # Sort the row centroids by x-coordinate
        row_centroids = row_centroids[np.argsort(row_centroids[:, 0])]

        # Calculate the average x-distance between centroids
        x_distances.append(np.diff(row_centroids[:, 0]))

    avg_x_dist = np.median(np.concatenate(x_distances))
    return int(np.round(avg_x_dist))


def get_median_y_distance(
    coords: np.ndarray,
    grid: np.ndarray,
    n_rows: Optional[int] = None
) -> int:
    """Calculate the median distance between centroids along the y-axis.

    Args:
        coords (np.ndarray): An array of coordinates.
        grid (np.ndarray): A binary grid.
        n_rows (int, optional): The number of rows in the grid.

    Returns:
        int: The median y-distance between centroids.

    """
    if n_rows is None:
        n_rows = count_rows(grid)

    row_width = grid.shape[1] // n_rows

    row_means = []
    for row in range(n_rows):
        # Define the y-limits for the current row
        y_min = row * row_width
        y_max = (row + 1) * row_width if row < n_rows - 1 else grid.shape[1]
        print("y_min: ", y_min, "y_max: ", y_max)
        # Get the grid within the y-limits
        row_centroids = coords[(coords[:, 1] >= y_min) & (coords[:, 1] < y_max)]
        # if len(row_centroids) == 0:
        #     avg_y = 30
        #     #n_rows -= 1
        #     #row_width = grid.shape[1] // n_rows
        # else:
        #     # Get the average y-value for the row
        print("row_centroids: ", row_centroids[:, 1])
        avg_y = np.mean(row_centroids[:, 1])
        print("avg_y: ", avg_y)
        row_means.append(avg_y)

    avg_y_dist = np.median(np.diff(row_means))
    print("avg_y_dist: ", avg_y_dist)
    return int(np.round(avg_y_dist))


def get_x_row_offset(
    grid: np.ndarray,
    max_offset: int = 100,
    n_rows: Optional[int] = None
) -> int:
    """Find the best x-offset for aligning rows.

    Args:
        grid (np.ndarray): A binary grid.
        max_offset (int): The maximum offset to search.
        n_rows (int, optional): The number of rows in the grid.

    Returns:
        int: The best x-row offset.

    """
    if n_rows is None:
        n_rows = count_rows(grid)

    row_width = grid.shape[1] // n_rows

    offset_scores = []
    offsets = list(range(max_offset))
    for offset in offsets:
        grid_zero = np.zeros(grid.shape)
        for row in range(n_rows):
            # Define the y-limits for the current row
            y_min = row * row_width
            y_max = (row + 1) * row_width if row < n_rows - 1 else grid.shape[1]

            # Get the grid within the y-limits
            grid_row = grid[:, y_min:y_max]

            # Get the coordinates of True values
            coords_row = np.argwhere(grid_row)

            # Offset these coordinates
            off_x = coords_row[:, 0] - (offset * row)

            # Truncate the x-coordinates to the grid size
            off_x = np.clip(off_x, 0, grid.shape[0] - 1)

            # Update grid_zero with the offset coordinates
            grid_zero[off_x, coords_row[:, 1]] = 1

        grid_sum = grid_zero.max(axis=1).sum()
        offset_scores.append(grid_sum)

    best_offset = offsets[np.argmin(offset_scores)]
    return best_offset


def detect_grid(coords: np.ndarray) -> "GridDefinition":
    from .segmentation import GridDefinition

    # Find the rotation angle
    angle = find_angle(coords)

    # Rotate the coordinates
    rotated_coords = rotate_coordinates(coords, angle, buffer=50)

    # Create an MxN array of rotated coordinates
    r_array = create_rotated_grid(coords, angle)

    # Find the number of rows in the grid
    n_rows = count_rows(r_array)

    # Get the median x-distance between centroids
    x_dist = get_median_x_distance(rotated_coords, r_array)

    # Get the median y-distance between rows
    y_dist = get_median_y_distance(rotated_coords, r_array)

    # Get the best x-row offset, indicating the offset of each row
    # relative to the previous row
    row_offset = get_x_row_offset(r_array, max_offset=x_dist)

    return GridDefinition(x_spacing=x_dist, y_spacing=y_dist, angle=angle, row_offset=row_offset)


def find_closest_to_centroid(points: np.ndarray) -> np.ndarray:
    """Find the point closest to the centroid of a set of points.

    Args:
        points (np.ndarray): An array of points, of shape (n, 2).

    Returns:
        np.ndarray: The point closest to the centroid.

    """
    # Convert points to a NumPy array if it's not already
    points = np.array(points)

    # Calculate the centroid
    centroid = np.mean(points, axis=0)

    # Calculate squared distances from each point to the centroid
    # We use squared distance to avoid unnecessary sqrt calculations
    distances = np.sum((points - centroid)**2, axis=1)

    # Find the index of the minimum distance
    closest_index = np.argmin(distances)

    # Return the closest point
    return points[closest_index]