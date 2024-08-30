import torch
import cv2
import os
import zipfile
import supervision as sv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cupy as cp
from typing import Dict, Tuple, List
from roifile import ImagejRoi
from attiicc import segment_anything as sa

class SamSegmenter:
    '''Segment nanowell images using a pretrained SAM model.
    
    This interface is designed to be used with nanowell images. 
    '''

    def __init__( 
        self,
        model_path: str = None,
        model_type: str = "vit_h",
        png_path: str = None,
        tif_path: str = None,
        **kwargs
    ) -> None:
        '''
        Initialize a SAM model and calculate the segmentation for an image.
        Inputs:
            model_path (str): The path to the model checkpoint. This must be downloaded \
                from Meta on a user's local machine. Checkpoints can be downloaded from \
                https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
            model_type (str, optional): Specify the sam model type to load.\
            Default is "vit_h". Can use "vit_b", "vit_l", or "vit_h".
            png_path (str, optional): The path to the image to segment. \
            Default is None.
            tif_path (str, optional): The path to the tif image to crop with \
                the ROIs. Default is None.
        Outputs:
            None
        '''
        assert isinstance(model_path, str), "Model checkpoint path on local machine must be specified for segmentation. \
            Model checkpoints must be downloaded from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints."
        self.model_path = model_path
        self.model_type = model_type
        self._png_path = png_path
        self._tif_path = tif_path
        self._sam_result = None
        self.img_bgr = None
        self.sam = None
        self.segmentation = None
        self.area = None
        self.bbox = None
        self.predicted_iou = None
        self.point_coords = None
        self.stability_score = None
        self.crop_box = None

        if png_path is not None:
            self.sam = self._load_sam_model(model_type=model_type)
            self._sam_result, self.img_bgr = self._segment_image(self.sam, self._png_path)
        if self._sam_result is not None:
            self.segmentation = [mask["segmentation"] for mask in self._sam_result]
            self.area = [mask["area"] for mask in self._sam_result]
            self.bbox = [mask["bbox"] for mask in self._sam_result]
            self.predicted_iou = [mask["predicted_iou"] for mask in self._sam_result]
            self.point_coords = [mask["point_coords"] for mask in self._sam_result]
            self.stability_score = [mask["stability_score"] for mask in self._sam_result]
            self.crop_box = [mask["crop_box"] for mask in self._sam_result]
    
    def update_image(self, 
                     png_path: str, 
                     tif_path: str) -> None:
        '''
        Update the image path and recalculate the segmentation results \
        without re-loading the SAM model. 
        '''
        self._png_path = png_path
        self._tif_path = tif_path
        self._sam_result, self.img_bgr = self._segment_image(self.sam, self._png_path)
        self.segmentation = [mask["segmentation"] for mask in self._sam_result]
        self.area = [mask["area"] for mask in self._sam_result]
        self.bbox = [mask["bbox"] for mask in self._sam_result]
        self.predicted_iou = [mask["predicted_iou"] for mask in self._sam_result]
        self.point_coords = [mask["point_coords"] for mask in self._sam_result]
        self.stability_score = [mask["stability_score"] for mask in self._sam_result]
        self.crop_box = [mask["crop_box"] for mask in self._sam_result]
        return
    
    def sam_area_filter(self, 
                        target_area: List[str]=[11100,12000]) -> None:
        '''
        Filter the SAM results by area.
        Inputs:
            target_area (list, optional): The target area of the ROI. \
                Default is [11500,13600] which is the area of the nanowells where \
                the first value is the lower bound and the second value is the upper bound.\
        '''
        self._sam_result = [mask for mask in self._sam_result if target_area[0] < mask['area'] < target_area[1]]
        self.segmentation = [mask["segmentation"] for mask in self._sam_result]
        self.area = [mask["area"] for mask in self._sam_result]
        self.bbox = [mask["bbox"] for mask in self._sam_result]
        self.predicted_iou = [mask["predicted_iou"] for mask in self._sam_result]
        self.point_coords = [mask["point_coords"] for mask in self._sam_result]
        self.stability_score = [mask["stability_score"] for mask in self._sam_result]
        self.crop_box = [mask["crop_box"] for mask in self._sam_result]

    def _load_sam_model(self, 
                        model_type: str) -> sa.Sam:
        '''
        Loads a pretrained SAM model.
        Input:
            model_type (str): Specify the sam model type to load.
            Can use "vit_b", "vit_l", or "vit_h". 
        Output:
            sam: The loaded SAM model.
        '''
        if torch.cuda.is_available():
            print("CUDA is available.")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        else:
            print("CUDA is not available.")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = model_type
        # Load the model
        sam = sa.sam_model_registry[MODEL_TYPE](checkpoint=self.model_path).to(device=DEVICE)
        print("Model Loaded")
        return sam


    def _segment_image(self, 
                       sam, 
                       png_path: str) -> Tuple[Dict, np.ndarray]:
        '''
        Segments an image using a pretrained SAM model.
        Inputs:
            sam (Sam): The loaded SAM model.
            png_path (str): The path to the image to segment.
        Outputs:
            sam_result (dict): The segmentation results. Contains: 
                `segmentation` : the mask 
                `area` : the area of the mask in pixels
                `bbox` : the boundary box of the mask in XYWH format
                `predicted_iou` : the model's own prediction for the quality of the mask
                `point_coords` : the sampled input point that generated this mask
                `stability_score` : an additional measure of mask quality
                `crop_box` : the crop of the image used to generate this mask in XYWH format
        '''
        mask_generator = sa.SamAutomaticMaskGenerator(sam)
        image_bgr = cv2.imread(png_path) # cv2 reads in BGR format
        print("PNG Path: ", png_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # convert to RGB
        sam_result = mask_generator.generate(image_rgb)
        return sam_result, image_bgr
    
    def plot_segmented_image(self, 
                             titles: List[str] = ['Source Image', 'Segmented Image'], 
                             save: bool = False, 
                             save_path: str = None, 
                             **kwargs) -> None:
        '''
        Plots the original image and the segmented image side-by-side.
        Inputs:
            titles (list, optional): The titles to use for the images. Default is ['Source Image', 'Segmented Image'].
            save (bool, optional): Whether to save the image. Default is False.
            save_path (str, optional): The path to save the image. Default is None.
        '''
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=self._sam_result)
        annotated_image = mask_annotator.annotate(scene=self.img_bgr.copy(), detections=detections)
        # Create a figure and a 1x2 grid of axes
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the images onto the axes
        axs[0].imshow(self.img_bgr)
        axs[0].set_title(titles[0])
        axs[0].axis('off')

        axs[1].imshow(annotated_image)
        axs[1].set_title(titles[1])
        axs[1].axis('off')
        if save:
        # Save the figure
            fig.savefig(save_path)
        # Display the figure
        plt.show()
        return 

    def plot_masks(self, 
                   save: bool = False, 
                   save_path: str = None, 
                   grid_size: Tuple[int,int] = (30, 5),
                   size: Tuple[int,int] = (48,96),
                   **kwargs) -> None:
        '''
        Plots masks from the segmentation results.
        Inputs:
            save (bool, optional): Whether to save the image. Default is False.
            save_path (str, optional): The path to save the image. Default is None.
            grid_size (tuple, optional): The grid size for the plot. Default is (9,10).
            size (tuple, optional): The size of the plot. Default is (48,96).
        Outputs:
            None
        '''
        masks = [
        mask['segmentation']
        for mask
        in sorted(self._sam_result, key=lambda x: x['area'], reverse=True)]
        num_images = len(masks)
        grid_size = grid_size
        # Create a figure and a grid of axes
        fig, axs = plt.subplots(*grid_size, figsize=size)
        # Reshape axs to 1-D array to easily iterate over
        axs = axs.ravel()
        # Plot the images onto the axes
        for i in range(num_images):
            axs[i].imshow(masks[i])
            axs[i].axis('off')
        # Remove unused subplots
        if num_images < np.prod(grid_size):
            for j in range(num_images, np.prod(grid_size)):
                fig.delaxes(axs[j])
            if save:
                plt.savefig(save_path)    
        return

    def _filter_duplicate_masks(self, 
               centroid_list_sorted: list, 
               coordinate_dict: dict, 
               filter_distance: int,
               roi_path: str = None, 
               save_heatmap: bool = False,
               validation_path: str = None) -> list:
        '''
        Filter duplicate ROIs based on the distance between the centroids.
        Helper function for generate_rois.
        '''
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

        # Removing coordinates from the filtered list
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
                image_name = os.path.basename(self._png_path).rstrip(".png")
                validation_dir = os.path.join(roi_path, "validation_plots")
                if not os.path.exists(validation_dir):
                    print("Making directory at: ", validation_dir)
                    os.makedirs(validation_dir)
                save_path = os.path.join(validation_dir, f"{image_name}_heatmap.png")
            else:
                save_path = os.path.join(validation_path, f"{image_name}_heatmap.png")

            plt.savefig(save_path)
            print(f"Heatmap saved to: {save_path}")
            plt.close()

        return centroid_list_filtered

    def generate_rois(self,
                      target_area: List[List[int]] = [11100,12000], 
                      filter_distance: int = 10,
                      roi_path: str = None,
                      roi_archive: bool = True,
                      validation_plot:bool = False,
                      print_plot:bool = False,
                      validation_path:bool = None,
                      save_heatmap: bool = False,
                      filter_duplicates: bool = False,
                      **kwargs) -> list:
        '''
        Generate ROIs from the segmentation results.
        Inputs:
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
        Outputs:
            roi_and_box_and_centroid_list: a list of lists containing the ROIs and the bounding boxes, sorted
                in order by the y-coordinate of the centroid, and the (X,Y) coordinates of the centroid. 
                The first list contains ROIs and the second list contains lists of the box coordinates in XYWH format.
                The third list is a list of tuples containing the centroid coordinates.
        '''
        image_name = os.path.basename(self._png_path).rstrip(".png")
        seg_num = 1
        centroid_list = []
        roi_list = []
        box_list = []
        centroid_coords_filtered = []
        coordinate_dict = {}
        for seg, box in zip(self.segmentation, self.bbox):
            binary_image = np.uint8(seg) * 255
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                M = cv2.moments(contour)
                if area > target_area[0]: # only select contours that are the nanowells (some small contours from cells may be present)
                    points = contour.squeeze().tolist()
                    roi = ImagejRoi.frompoints(points)
                    if M["m00"] != 0: # calculate the centroid to allow filtering of duplicate nanowells
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        coords_id = [cX, cY]
                        coordinate_dict[(cX, cY)] = [roi, seg_num, box, (cX, cY)]
                        centroid_list.append(coords_id)
            seg_num += 1
        # Sort the list of lists by the value at index 0
        centroid_list_sorted = sorted(centroid_list, key=lambda x: x[0])
        if filter_duplicates is True:
            # Remove duplicates
            print("Total number of ROIs before filtering: ", len(centroid_list_sorted))
            print("Filtering for duplicates...")
            filtered_coordinates = self._filter_duplicate_masks(centroid_list_sorted,
                                                coordinate_dict,
                                                filter_distance=filter_distance,
                                                roi_path=roi_path,
                                                save_heatmap=save_heatmap,
                                                validation_path=validation_path)
            # Sort the list by y-coordinate
        else: 
            filtered_coordinates = centroid_list_sorted
        filtered_coordinates = sorted(filtered_coordinates, key=lambda x: x[1])
        for i in filtered_coordinates:
            roi_list.append(coordinate_dict[tuple(i)][0])
            box_list.append(coordinate_dict[tuple(i)][2])
            centroid_coords_filtered.append(coordinate_dict[tuple(i)][3])
            
        roi_and_box_and_centroid_list = [roi_list, box_list, centroid_coords_filtered]
        if roi_path is not None:
            print("Saving ROIs to: ", roi_path+'/'+image_name)
            new_path = os.path.join(roi_path, image_name)
            if not os.path.exists(new_path):
                print("Making directory at: ", new_path)
                os.makedirs(new_path)
            for i, j in enumerate(filtered_coordinates):
                roi = coordinate_dict[tuple(j)][0]
                roi_name = f"{image_name}_ROI_{i+1}.roi"
                roi.tofile(os.path.join(new_path, roi_name))
            print(f"ROIs saved for {image_name}")
            if roi_archive:
                print("Archiving ROIs to: ", f'{roi_path}/{image_name}_roi.zip')
                with zipfile.ZipFile(f'{roi_path}/{image_name}_rois.zip', 'w') as zipf:
                    for root, _, files in os.walk(new_path):
                        for file in files:
                            zipf.write(os.path.join(root, file), file)
            if validation_plot:
                # Load image
                img = mpimg.imread(self._png_path)
                print("Generating validation plot for ", self._png_path)
                plt.imshow(img, cmap='gray')
                # Create a scatter plot of the centroids
                plt.scatter(*zip(*filtered_coordinates), color='yellow', marker='o')
                plt.title(f"Centroids for {image_name}")
                # Annotate each point with its label
                # TODO: Modify the label to be the assigned well name.
                for (x, y), i in zip(filtered_coordinates, range(len(filtered_coordinates))):
                    plt.text(x, y, str(i), color='white')
                if validation_path is None:
                    validation_dir = os.path.join(roi_path, "validation_plots")
                    if not os.path.exists(validation_dir):
                        print("Making directory at: ", validation_dir)
                        os.makedirs(validation_dir)
                    plt.savefig(os.path.join(validation_dir, f"{image_name}_validation.png"))
                else:
                    plt.savefig(os.path.join(validation_path, f"{image_name}_validation.png"))
                if print_plot:
                    plt.show()
                plt.close()
        return roi_and_box_and_centroid_list
    