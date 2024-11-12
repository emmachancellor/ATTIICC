import cv2
import os
import zipfile
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import math
from PIL import Image
from os.path import join, exists
from scipy.spatial import cKDTree

from typing import Dict, Tuple, List, Optional
from .utils import _get_filename_without_ext
from .contours import (
    calculate_average_contour, detect_grid, find_closest_to_centroid, rotate_coordinates
)

# -----------------------------------------------------------------------------

class GridDefinition:
    """Class to store the definition of a grid schematic.

    The grid definition includes the x and y spacing between wells, the angle
    of the grid, the row offset (how far to the right each row is offset
    from the previous row), and optionally, the shape of the wells.

    """

    def __init__(
        self,
        x_spacing: int,
        y_spacing: int,
        angle: float,
        row_offset: int,
        shape_contours: Optional[np.ndarray] = None
    ) -> None:
        """Initialize the grid definition.

        Args:
            x_spacing (int): The spacing between wells in the x-direction.
            y_spacing (int): The spacing between wells in the y-direction.
            angle (float): The angle of the grid.
            row_offset (int): The offset of each row.
            shape_contours (np.ndarray, optional): The shape contours of the wells.
                Default is None.

        """
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.angle = angle
        self.row_offset = row_offset
        shape_contours = np.array(shape_contours) if shape_contours is not None else None
        self.shape_contours = shape_contours

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"<GridDefinition(\n  x_spacing={self.x_spacing},\n  y_spacing={self.y_spacing},\n  angle={self.angle},\n  row_offset={self.row_offset},\n  has_shape={self.shape_contours is not None}\n)>"

    def to_dict(self):
        """Convert the grid definition to a dictionary."""
        return {
            "x_spacing": self.x_spacing,
            "y_spacing": self.y_spacing,
            "angle": self.angle,
            "row_offset": self.row_offset,
            "shape_contours": self.shape_contours
        }

    @classmethod
    def from_dict(cls, grid_dict):
        """Create a grid definition from a dictionary."""
        return cls(
            x_spacing=grid_dict["x_spacing"],
            y_spacing=grid_dict["y_spacing"],
            angle=grid_dict["angle"],
            row_offset=grid_dict["row_offset"],
            shape_contours=(np.array(grid_dict["shape_contours"])
                            if grid_dict["shape_contours"] is not None else None)
        )

    @classmethod
    def load(cls, path: str) -> "GridDefinition":
        """Load a grid definition from a .npz file."""
        data = np.load(path)
        return cls.from_dict(data)

    @property
    def has_shape(self) -> bool:
        """Check if the grid definition has shape contours."""
        return self.shape_contours is not None

    def save(self, path: str) -> None:
        """Save the grid definition to a .npz file."""
        np.savez(path, **self.to_dict())

    def set_shape(self, shape_contours: np.ndarray, rotate: bool = False) -> None:
        """Set the shape contours of the grid.

        Args:
            shape_contours (list): A 2D array of coordinates indicating the shape contour.
            rotate (bool, optional): Whether to rotate the shape contour. Default is False.

        """
        # Convert the shape contours to a numpy array
        shape_contours = np.array(shape_contours)
        # Rotate the shape contours if necessary
        if rotate:
            self.shape_contours = rotate_coordinates(shape_contours, self.angle)
        else:
            self.shape_contours = shape_contours

    def build(
        self,
        anchor: Tuple[int, int],
        x_max: int,
        y_max: int,
        img: Optional[np.ndarray] = None
    ) -> "Plate":
        """Build a grid of wells from the grid definition.

        Args:
            anchor (tuple): The anchor point of the grid.
            x_max (int): The maximum x-coordinate of the image.
            y_max (int): The maximum y-coordinate of the image.
            img (np.ndarray, optional): The image to plot the grid on. Default is None.

        Returns:
            Plate: A collection of Well objects.

        """
        # Unpack the anchor
        anchor_x, anchor_y = anchor

        # Keep track of well centroids
        centroids = []

        # Starting at the anchor (center), build out the grid centroids
        # in both the positive and negative directions.

        # First, let's build the negative rows
        n_rows_neg = int(math.ceil(float(anchor_y) / self.y_spacing))
        for i in range(n_rows_neg):
            y = anchor_y - i * self.y_spacing
            row_offset = self.row_offset * i
            row_anchor_x = anchor_x - row_offset

            # Positive x direction
            n_pos_cols = int(math.ceil(float(x_max - row_anchor_x) / self.x_spacing)) + 1
            for j in range(n_pos_cols):
                x = row_anchor_x + (j * self.x_spacing)
                centroids.append((x, y))

            # Negative x direction
            n_neg_cols = int(math.ceil(float(row_anchor_x) / self.x_spacing)) + 1
            for j in range(n_neg_cols):
                x = row_anchor_x - (j * self.x_spacing)
                centroids.append((x, y))

        # Now, let's build the positive rows
        n_rows_pos = int(math.ceil(float(y_max - anchor_y) / self.y_spacing))
        for i in range(n_rows_pos):
            y = anchor_y + i * self.y_spacing
            row_offset = self.row_offset * i
            row_anchor_x = anchor_x + row_offset

            # Positive x direction
            n_pos_cols = int(math.ceil(float(x_max - row_anchor_x) / self.x_spacing)) + 1
            for j in range(n_pos_cols):
                x = row_anchor_x + (j * self.x_spacing)
                centroids.append((x, y))

            # Negative x direction
            n_neg_cols = int(math.ceil(float(row_anchor_x) / self.x_spacing)) + 1
            for j in range(n_neg_cols):
                x = row_anchor_x - (j * self.x_spacing)
                centroids.append((x, y))

        # Rotate the centroids around the anchor and remove duplicates.
        centroids = np.array(centroids)
        centroids = np.unique(centroids, axis=0)
        if self.angle != 0:
            centroids = rotate_coordinates(centroids, angle=-1 * self.angle, anchor=anchor)

        # Finally, filter out any centroids that are outside the image bounds
        centroids = [tuple(c) for c in centroids if 0 <= c[0] < x_max and 0 <= c[1] < y_max]

        # Build the Plate object
        rotated_shape = rotate_coordinates(self.shape_contours, angle=-1 * self.angle)
        wells = [Well(rotated_shape, centroid, img=img) for centroid in centroids]
        return Plate(*wells, img=img, grid_definition=self)


class Well:
    """Class to store information about a single well on a plate."""

    def __init__(
        self,
        roi: np.ndarray,
        centroid: Tuple[int,int],
        box: Optional[Tuple[int,int,int,int]] = None,
        *,
        img: Optional[np.ndarray] = None,
    ) -> None:
        """Build a Well object, representing a single well on a larger plate.

        Args:
            roi (np.ndarray): The ROI object.
            box (tuple): The boundary box of the ROI in XYWH format.
            centroid (tuple): The centroid of the ROI.

        Keyword Args:
            img (np.ndarray, optional): The underlying image. Default is None.

        """
        self.roi = roi
        self.centroid = centroid
        if box is None:
            box = cv2.boundingRect(roi)
        self.box = box
        self.img = img

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"<Well object with centroid {self.centroid}>"

    def plot(self,
             save_path: Optional[str] = None):
        """Plot the well."""
        # Plot the ROI
        plt.plot(*zip(*self.roi), color='black')

        # Plot the centroid
        plt.scatter(*self.centroid, color='red', marker='o')

        # Ensure the plot is square
        plt.gca().set_aspect('equal', adjustable='box')

        # Ensure the origin is in the top left
        plt.gca().invert_yaxis()

        # Save the plot
        if save_path:
            plt.savefig(save_path)

    def roi_to_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert the ROI to a binary mask.

        Args:
            image_shape (tuple): Shape of the mask (height, width).

        Returns:
            mask (np.ndarray): Binary mask of the ROI.
        """
        # Convert ROI relative coordinates to absolute coordinates by adding centroid offset
        roi_absolute = self.roi + np.array(self.centroid)

        # Create a blank mask of the image size
        mask = np.zeros(image_shape[:2], dtype=np.uint8)

        # Fill the mask with the ROI using cv2.fillPoly
        cv2.fillPoly(mask, [roi_absolute.astype(np.int32)], 255)

        return mask

    def get_image(self) -> Image:
        """Extract the region of the image corresponding to the ROI.

        Returns:
            extracted_image (np.ndarray): Image cropped to the ROI.
        """
        if self.img is None:
            raise ValueError("Image data is missing in the Well object.")

        # Create a mask from the ROI
        mask = self.roi_to_mask(self.img.shape)

        # Apply the mask to the image
        extracted_image = cv2.bitwise_and(self.img, self.img, mask=mask)

        # Crop the image using the bounding box
        x, y, w, h = self.box
        x += self.centroid[0]
        y += self.centroid[1]
        cropped_image = extracted_image[y:y + h, x:x + w]

        image = Image.fromarray(cropped_image)

        return image


class Plate:

    def __init__(
        self,
        *wells: Well,
        img: Optional[np.ndarray] = None,
        grid_definition: Optional[GridDefinition] = None
    ) -> None:
        """Build a collection of Wells.

        Args:
            *wells (Well): A collection of Well objects.

        Keyword Args:
            img (np.ndarray, optional): The underlying image. Default is None.
            grid_definition (GridDefinition, optional): The grid definition.
                Default is None.

        """
        self.wells = wells
        self.img = img
        self.grid_definition = grid_definition

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"<Plate object with {len(self.wells)} wells>"

    def __len__(self):
        return len(self.wells)

    def __getitem__(self, idx):
        return self.wells[idx]

    def __iter__(self):
        return iter(self.wells)

    def __next__(self):
        return next(self.wells)

    def __contains__(self, item):
        return item in self.wells

    def __add__(self, other):
        return Plate(*(self.wells + other.wells), img=self.img)

    def __sub__(self, other):
        return Plate(*(r for r in self.wells if r not in other.wells), img=self.img)

    def __eq__(self, other):
        return self.wells == other.wells

    def __ne__(self, other):
        return self.wells != other.wells

    def pop(self, idx):
        return self.wells.pop(idx)

    @property
    def centroids(self) -> List[Tuple[int,int]]:
        return np.array([r.centroid for r in self.wells])

    def remove_edge_wells(self, threshold: float = 0.9) -> None:
        """Remove wells at the edge of the image based on
        the percentage of the well that is inside the image bounds.

        Args:
            threshold (float, optional): The threshold for the percentage of
                the well inside the image bounds. Default is 0.9.

        """
        # For each well, find what percentage of the well is inside the image bounds
        img_h, img_w = self.img.shape[:2]
        orig_wells = self.wells
        wells = []
        for well in self.wells:
            # Get the bounding box
            x, y, w, h = well.box

            # Adjust for the image centroid
            x += well.centroid[0]
            y += well.centroid[1]

            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
            area = (x2 - x1) * (y2 - y1)
            well_area = w * h
            if area / well_area > threshold:
                wells.append(well)
            else:
                print("Removing edge well:", well.centroid)
        self.wells = wells
        print("Removed", len(orig_wells) - len(wells), "edge wells.")

    def plot(
        self,
        img: Optional[np.ndarray] = None,
        *,
        show_labels: bool = True,
        show_image: bool = True,
        show_contours: bool = True,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None
    ):
        """Plot the wells on the image."""
        if img is None:
            img = self.img

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the well centroids.
        if len(self.wells) > 0:
            ax.scatter(*zip(*[r.centroid for r in self.wells]), color='yellow', marker='o')

        # Plot the reference image.
        if show_image and img is not None:
            ax.imshow(img)

        # Plot the well contours.
        if show_contours:
            for well in self.wells:
                ax.plot(*zip(*(well.roi + well.centroid)), color='red')

        # Show labels
        if show_labels:
            for i, well in enumerate(self.wells):
                x, y = well.centroid
                ax.text(x, y, str(i), color='white')
        
        # Save the plot
        if save_path:
            plt.savefig(save_path)

    def get_average_contour(self, **kwargs) -> np.ndarray:
        """Get the average contour of the wells."""
        contours = [r.roi for r in self.wells]
        return calculate_average_contour(contours, **kwargs)

    def plot_average_contour(self, **kwargs):
        """Plot the average contour of the wells."""
        average_contour = self.get_average_contour(**kwargs)
        plt.plot(*zip(*average_contour), color='black')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()

    def detect_grid(self) -> GridDefinition:
        # Detect the well spacing and orientation
        grid_def = detect_grid(self.centroids)

        # Find the average contour
        contours = self.get_average_contour()

        # Apply the contours to the grid definition
        grid_def.set_shape(contours, rotate=True)

        return grid_def

    def apply_grid(self, grid: GridDefinition) -> "Plate":
        """Convert this collection into a grid of wells."""
        # First, find the anchor point,
        # which will be the point nearest to the center.
        anchor = find_closest_to_centroid(self.centroids)

        # Then, build the grid.
        return grid.build(anchor, x_max=self.img.shape[1], y_max=self.img.shape[0], img=self.img)

    def build_grid(self) -> "Plate":
        """Build a grid of wells."""
        grid_def = self.detect_grid()
        return self.apply_grid(grid_def)


class PlateStack:
    """Class to store a stack of well grids.

    A PlateStack is a collection of Plate objects, each representing a grid of wells.
    The PlateStack object can be used to store multiple grids of wells, such as those
    obtained from different time points or fields of view.

    """

    def __init__(
        self,
        *plates: Plate,
        img: Optional[np.ndarray] = None
    ) -> None:
        """Build a stack of well grids.

        Args:
            *plates (Plate): A collection of Plate objects.

        Keyword Args:
            img (np.ndarray, optional): The image to plot the grid on.

        """
        self.plates = plates
        if img is None:
            img = plates[0].img
        self._build_stack()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"<PlateStack object with {len(self.plates)} plates>"

    def __len__(self):
        return len(self.plates)

    def __getitem__(self, idx):
        return self.plates[idx]

    def _build_stack(self):
        """Build a stack of wells."""

        # Find the indices that are common to all grids.
        reference_grid = self.plates[0]
        common_indices = self._get_matching_well_indices(reference_grid, self.plates[1:])

        # In the reference well, remove the wells that are not common to all grids.
        reference_grid.wells = [reference_grid.wells[i] for i in common_indices]

        # In the subsequent wells, remove the wells that are not common to all grids.
        tree = cKDTree(reference_grid.centroids)
        for plate in self.plates[1:]:
            distances, ref_indices = tree.query(plate.centroids, k=1)

            # Remove indices that are very far away
            max_dist = reference_grid.grid_definition.x_spacing // 2
            idx_to_keep = np.where(distances < max_dist)[0]

            # Reorder the wells to match the reference grid
            idx_to_keep = sorted(idx_to_keep, key=lambda x: ref_indices[x])
            plate.wells = [plate.wells[i] for i in idx_to_keep]

            assert len(plate.wells) == len(reference_grid.wells), "The number of wells in the grids do not match."


    @staticmethod
    def _get_matching_well_indices(
        reference: Plate,
        plates: List[Plate]
    ) -> List[int]:
        """Find the indices of the wells that are common to all plates."""
        # Build a KDTree from the reference grid.
        tree = cKDTree(reference.centroids)

        # For each subsequent grid, find the nearest neighbor in the reference grid.
        matched_well_idx = [np.arange(len(reference.centroids))]
        for plate in plates:
            tree = cKDTree(plate.centroids)
            distances, indices = tree.query(plate.centroids, k=1)

            # Remove indices that are very far away
            max_dist = reference.grid_definition.x_spacing // 2
            indices = indices[distances < max_dist]

            matched_well_idx.append(indices.flatten())

        # Find the indices that are common to all grids.
        common_indices = set.intersection(*[set(indices) for indices in matched_well_idx])

        return list(common_indices)

    def plot_well(self, well_idx: int):
        """Plot a specific well from a specific grid."""
        if well_idx >= len(self.plates[0]):
            raise ValueError(f"Well index {well_idx} is out of range.")

        fig, ax = plt.subplots(1, len(self.plates), figsize=(5 * len(self.plates), 5))
        for i in range(len(self.plates)):
            well = self.plates[i][well_idx]
            ax[i].imshow(well.get_image())
            ax[i].set_title(f"Time Point {i}")
            ax[i].axis('off')
    
    def save_all_wells(self, save_dir: str):
        """Save a specific well from a specific grid."""
        if not exists(save_dir):
            os.makedirs(save_dir)   
        for well_idx in range(len(self.plates[0])):
            for i in range(len(self.plates)):
                well = self.plates[i][well_idx]
                image = well.get_image()
                save_path = save_dir + f"/well_{well_idx}_time_{i}.png"
                image.save(join(save_dir, f"/well_{well_idx}_time_{i}.png"))


# -----------------------------------------------------------------------------

class Segmentation:
    """Class to store the results of a segmentation model.

    The Segmentation object stores the masks generated by a segmentation model,
    along with additional information such as the area, bounding box, and centroid
    of each mask. The object can be used to filter masks based on area, plot the
    masks, and generate ROIs from the masks.

    """

    def __init__(
        self,
        masks: List[Dict],
        img: np.ndarray,
        *,
        image_path: str = None
    ) -> None:
        """Build a SAM Segmentation result object.

        Masks is expected to be a list of dictionaries containing:
            `segmentation` : the mask
            `area` : the area of the mask in pixels
            `bbox` : the boundary box of the mask in XYWH format
            `predicted_iou` : the model's own prediction for the quality of the mask
            `point_coords` : the sampled input point that generated this mask
            `stability_score` : an additional measure of mask quality
            `crop_box` : the crop of the image used to generate this mask in XYWH format

        """
        self.masks = masks
        self.img = img
        self.image_path = image_path

    @property
    def segmentation(self) -> List[np.ndarray]:
        return [mask["segmentation"] for mask in self.masks]

    @property
    def area(self) -> List[int]:
        return [mask["area"] for mask in self.masks]

    @property
    def bbox(self) -> List[Tuple[int,int,int,int]]:
        return [mask["bbox"] for mask in self.masks]

    @property
    def predicted_iou(self) -> List[float]:
        return [mask["predicted_iou"] for mask in self.masks]

    @property
    def point_coords(self) -> List[Tuple[int,int]]:
        return [mask["point_coords"] for mask in self.masks]

    @property
    def stability_score(self) -> List[float]:
        return [mask["stability_score"] for mask in self.masks]

    @property
    def crop_box(self) -> List[Tuple[int,int,int,int]]:
        return [mask["crop_box"] for mask in self.masks]

    @property
    def name(self):
        if self.image_path:
            return _get_filename_without_ext(self.image_path)
        else:
            return 'Unknown'

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"<Segmentation object with {len(self.masks)} masks>"

    def area_filter(self, target_area: Tuple[int,int] = (11100, 12000)) -> None:
        """Filter the SAM results by area.

        Args:
            target_area (list, optional): The target area of the ROI.
                Default is [11500,13600] which is the area of the nanowells where
                the first value is the lower bound and the second value is the upper bound.

        """
        self.masks = [mask for mask in self.masks if target_area[0] < mask['area'] < target_area[1]]

    def plot(
        self,
        titles: List[str] = ['Source Image', 'Segmented Image'],
        *,
        save: bool = False,
        save_path: str = None
    ) -> None:
        """Plot the original image and the segmented image side-by-side.

        Args:
            titles (list, optional): The titles to use for the images. Default is ['Source Image', 'Segmented Image'].
            save (bool, optional): Whether to save the image. Default is False.
            save_path (str, optional): The path to save the image. Default is None.

        """
        # Annotate the image
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=self.masks)
        annotated_image = mask_annotator.annotate(scene=self.img.copy(), detections=detections)

        # Create a figure and a 1x2 grid of axes
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the images onto the axes
        axs[0].imshow(self.img)
        axs[0].set_title(titles[0])
        axs[0].axis('off')

        axs[1].imshow(annotated_image)
        axs[1].set_title(titles[1])
        axs[1].axis('off')

        # Save the figure
        if save:
            fig.savefig(save_path)


    def plot_masks(
        self,
        save_path: str = None,
        *,
        grid_size: Tuple[int,int] = (30, 5),
        size: Tuple[int,int] = (48,96)
    ) -> None:
        """Plot masks from the segmentation results.

        Args:
            save_path (str, optional): The path to save the image. Default is None.
            grid_size (tuple, optional): The grid size for the plot. Default is (9,10).
            size (tuple, optional): The size of the plot. Default is (48,96).

        Returns:
            None
        """
        masks = [
            mask['segmentation']
            for mask in sorted(self.masks, key=lambda x: x['area'], reverse=True)
        ]

        # Create a figure and a grid of axes
        fig, axs = plt.subplots(*grid_size, figsize=size)

        # Reshape axs to 1-D array to easily iterate over
        axs = axs.ravel()

        if len(masks) > len(axs):
            raise ValueError(
                f"Number of masks ({len(masks)}) exceeds number of subplots ({len(axs)}). "
                "Please set grid_size to a larger value."
            )

        # Plot the images onto the axes
        for i in range(len(masks)):
            axs[i].imshow(masks[i])
            axs[i].axis('off')

        # Remove unused subplots
        if len(masks) < np.prod(grid_size):
            for j in range(len(masks), np.prod(grid_size)):
                fig.delaxes(axs[j])
            if save_path:
                plt.savefig(save_path)


    def _filter_duplicate_masks(
        self,
        centroid_list_sorted: list,
        coordinate_dict: dict,
        filter_distance: int,
        roi_path: str = None,
        save_heatmap: bool = False,
        validation_path: str = None
    ) -> list:
        """Filter duplicate ROIs based on the distance between the centroids.

        Helper function for generate_rois.

        Args:
            centroid_list_sorted (list): A list of centroids sorted in a specific order.
            coordinate_dict (dict): A dictionary mapping coordinates to segment numbers.
            filter_distance (int): The maximum distance allowed between centroids for them to be considered duplicates.
            roi_path (str, optional): The path to the ROI directory. Defaults to None.
            save_heatmap (bool, optional): Whether to save a heatmap of the centroid distances. Defaults to False.
            validation_path (str, optional): The path to the validation directory. Defaults to None.

        Returns:
            list: A list of filtered centroids without duplicates.

        """
        print("Filtering for duplicates...")
        print(f"Initial number of ROIs: {len(centroid_list_sorted)}")
        print(f"Filter distance: {filter_distance}")

        # Remove exact duplicates first
        unique_centroids = {}
        for i, centroid in enumerate(centroid_list_sorted):
            centroid_tuple = tuple(centroid)
            if centroid_tuple in unique_centroids:
                print(f"Removing exact duplicate centroid at index {i}: {centroid}")
            else:
                unique_centroids[centroid_tuple] = i

        centroid_list_unique = [centroid_list_sorted[i] for i in unique_centroids.values()]
        print(f"Number of ROIs after removing exact duplicates: {len(centroid_list_unique)}")

        # Compute distance matrix
        matrix_coordinates = cp.array(centroid_list_unique)
        difference = matrix_coordinates[:, cp.newaxis, :] - matrix_coordinates[cp.newaxis, :, :]
        sq_difference = cp.square(difference)
        distance_matrix = cp.sqrt(cp.sum(sq_difference, axis=2))

        # Create a mask for pairs within filter_distance
        mask = distance_matrix <= filter_distance

        # Create sets to keep track of ROIs to remove and preserve
        remove_coords = set()
        preserve_coords = set()

        n = len(centroid_list_unique)
        for i in range(n):
            if i in remove_coords:
                continue
            for j in range(i + 1, n):
                if j in remove_coords:
                    continue
                if mask[i, j]:
                    coord_i = tuple(centroid_list_unique[i])
                    coord_j = tuple(centroid_list_unique[j])
                    seg_num_i = coordinate_dict[coord_i][1]
                    seg_num_j = coordinate_dict[coord_j][1]
                    distance = distance_matrix[i, j].get()

                    print(f"Potential duplicate: {coord_i} and {coord_j}, distance: {distance:.2f}, seg_nums: {seg_num_i}, {seg_num_j}")

                    if distance < 1e-6:  # Consider as exact duplicate if distance is very small
                        remove_coords.add(j)
                        preserve_coords.add(i)
                        print(f"Removing exact duplicate ROI at index {j} (seg_num: {seg_num_j})")
                    elif seg_num_i != seg_num_j:
                        if i not in preserve_coords:
                            remove_coords.add(i)
                            preserve_coords.add(j)
                            print(f"Removing close ROI at index {i} (seg_num: {seg_num_i})")
                        else:
                            remove_coords.add(j)
                            print(f"Removing close ROI at index {j} (seg_num: {seg_num_j})")
                    else:
                        preserve_coords.add(i)
                        preserve_coords.add(j)
                        print(f"Preserving ROIs at indices {i} and {j} due to same seg_num: {seg_num_i}")

        # Remove coordinates from the filtered list
        if len(remove_coords) > 0:
            print(f"Removing {len(remove_coords)} coordinates: {remove_coords}")
        else:
            print("No additional coordinates to remove.")

        centroid_list_filtered = [x for i, x in enumerate(centroid_list_unique) if i not in remove_coords]

        print(f"Final number of ROIs: {len(centroid_list_filtered)} \n")

        # Save heatmap
        if save_heatmap:
            plt.figure(figsize=(12, 10))
            plt.imshow(distance_matrix.get(), cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Distance')
            plt.title('Centroid Distance Heatmap')
            plt.xlabel('Centroid Index')
            plt.ylabel('Centroid Index')

            # Highlight removed ROIs
            for idx in remove_coords:
                plt.axhline(y=idx, color='r', linestyle='--', alpha=0.5)
                plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)

            if validation_path is None:
                validation_dir = os.path.join(roi_path, "validation_plots")
                if not os.path.exists(validation_dir):
                    print("Making directory at: ", validation_dir)
                    os.makedirs(validation_dir)
                save_path = os.path.join(validation_dir, f"{self.name}_heatmap.png")
            else:
                save_path = os.path.join(validation_path, f"{self.name}_heatmap.png")

            plt.savefig(save_path)
            print(f"Heatmap saved to: {save_path}")
            plt.close()

        return centroid_list_filtered

    @staticmethod
    def _get_contours(img: np.ndarray) -> List[np.ndarray]:
        """Get the contours of a binary image."""
        binary_image = np.uint8(img) * 255
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_wells(
        self,
        min_area: int = 11100,
        filter_distance: int = 10,
        roi_path: str = None,
        roi_archive: bool = True,
        validation_path: bool = None,
        save_heatmap: bool = False,
        filter_duplicates: bool = False,
    ) -> Plate:
        """
        Generate ROIs from the segmentation results.

        Args:
            target_area (int, optional): The target area of the ROI. Default is 11500.
            filter_distance (int, optional): When filtering for duplicate ROIs
                this method will search for ROI centroids that are within +/-
                a number of pixels (similarity_filter). Default is 10 pixels.
            roi_path (str, optional): The path to a directory where ROIs can be saved. Default is None.
            roi_archive (bool, optional): Whether to save the ROIs as a .zip. Default is True.
                This will save the roi.zip file in the roi_path. Default is True.
            validation_plot (bool, optional): Whether to save the validation plot. Default is False.
            print_plot (bool, optional): Whether to print the validation plot. Default is False.
            validation_path (str, optional): Path to a directory to save validation plots
            save_heatmap (bool, optional): Whether to save a correlation heatmap of the centroid coordinates.

        Returns:
            Plate containing segmentation Well objects, sorted in order by the y-coordinate of the centroid, and the (X,Y) coordinates of the centroid.

        """
        centroid_list = []
        regions = {}

        for seg, box in zip(self.segmentation, self.bbox):
            contours = self._get_contours(seg)
            for contour in contours:
                # Only select contours that are the nanowells (some small contours from cells may be present)
                if cv2.contourArea(contour) > min_area:
                    # Calculate the centroid of the contour
                    M = cv2.moments(contour)
                    points = contour.squeeze()

                    # Calculate the centroid to allow filtering of duplicate nanowells
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centroid = (cX, cY)
                        regions[centroid] = Well(points - np.array(centroid), centroid, box, img=self.img)
                        centroid_list.append(centroid)

        # Sort the list of lists by the value at index 0 (x)
        centroid_list = sorted(centroid_list, key=lambda x: x[0])

        # Remove duplicates
        if filter_duplicates:
            centroid_list = self._filter_duplicate_masks(
                centroid_list,
                regions,
                filter_distance=filter_distance,
                roi_path=roi_path,
                save_heatmap=save_heatmap,
                validation_path=validation_path
            )

        # Sort the list of lists by the value at index 1 (y)
        centroid_list = sorted(centroid_list, key=lambda x: x[1])

        # Create a list of ROIs, box coordinates, and centroid coordinates
        regions_to_return = [regions[coord] for coord in centroid_list]

        # Export ROIs as ImageJ and archive in ZIP format.
        if roi_path is not None:
            try:
                from roifile import ImagejRoi
            except ImportError:
                raise ImportError("The roifile package is required to export ROIs. Please install it using 'pip install roifile'.")

            # Create an export directory based on the image name
            dest = join(roi_path, self.name)
            if not exists(dest):
                os.makedirs(dest)

            # Export the ImageJ ROI files
            for i, j in enumerate(centroid_list):
                # Create an ImagejRoi object from the contour points
                coords = regions[tuple(j)].roi
                roi = ImagejRoi.frompoints(coords)

                roi_name = f"{self.name}_ROI_{i+1}.roi"
                roi.tofile(join(dest, roi_name))

            print(f"ROIs saved at {dest}")

            # Archive the ROIs into a zip file
            if roi_archive:
                print("Archiving ROIs to: ", f'{roi_path}/{self.name}_roi.zip')
                with zipfile.ZipFile(f'{roi_path}/{self.name}_rois.zip', 'w') as zipf:
                    for root, _, files in os.walk(dest):
                        for file in files:
                            zipf.write(os.path.join(root, file), file)

        return Plate(*regions_to_return, img=self.img)

