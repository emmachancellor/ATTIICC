import cv2
import os
import zipfile
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from os.path import join, exists

from typing import Dict, Tuple, List, Optional
from roifile import ImagejRoi
from .utils import _get_filename_without_ext

# -----------------------------------------------------------------------------

class Region:

    def __init__(
        self,
        roi: ImagejRoi,
        box: Tuple[int,int,int,int],
        centroid: Tuple[int,int]
    ) -> None:
        """Build a SAM Region object.

        Args:
            roi (ImagejRoi): The ROI object.
            box (tuple): The boundary box of the ROI in XYWH format.
            centroid (tuple): The centroid of the ROI.

        """
        self.roi = roi
        self.box = box
        self.centroid = centroid

class RegionCollection:

    def __init__(
        self,
        *regions: Region,
        img: Optional[np.ndarray] = None
    ) -> None:
        """Build a SAM RegionCollection object.

        Args:
            *regions (Region): A collection of Region objects.

        """
        self.regions = regions
        self.img = img

    def plot(
        self,
        img: Optional[np.ndarray] = None,
        show_labels: bool = True
    ):
        """Plot the regions on the image."""
        if img is None:
            img = self.img

        # Plot the region centroids.
        plt.scatter(*zip(*[r.centroid for r in self.regions]), color='yellow', marker='o')

        # Plot the reference image.
        if img is not None:
            plt.imshow(img)

        # Show labels
        if show_labels:
            for i, region in enumerate(self.regions):
                x, y = region.centroid
                plt.text(x, y, str(i), color='white')

        plt.show()

# -----------------------------------------------------------------------------

class Segmentation:
    def __init__(
        self,
        masks: List[Dict],
        img: np.ndarray,
        *,
        image_path: str = None,
        tif_path: str = None
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

        plt.show()

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

        plt.show()

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

    def generate_rois(
        self,
        min_area: int = 11100,
        filter_distance: int = 10,
        roi_path: str = None,
        roi_archive: bool = True,
        validation_path: bool = None,
        save_heatmap: bool = False,
        filter_duplicates: bool = False,
    ) -> RegionCollection:
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
            RegionCollection containing segmentation Region objects, sorted in order by the y-coordinate of the centroid, and the (X,Y) coordinates of the centroid.

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
                    points = contour.squeeze().tolist()

                    # Create an ImagejRoi object from the contour points
                    roi = ImagejRoi.frompoints(points)

                    # Calculate the centroid to allow filtering of duplicate nanowells
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        coords_id = (cX, cY)
                        regions[coords_id] = Region(roi, box, coords_id)
                        centroid_list.append(coords_id)

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
            # Create an export directory based on the image name
            dest = join(roi_path, self.name)
            if not exists(dest):
                os.makedirs(dest)

            # Export the ImageJ ROI files
            for i, j in enumerate(centroid_list):
                roi = regions[tuple(j)].roi
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

        return RegionCollection(*regions_to_return, img=self.img)

