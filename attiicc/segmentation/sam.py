import numpy as np
import torch
import cv2
import os
import supervision as sv
import matplotlib.pyplot as plt
import zipfile
from typing import Dict, Tuple
from roifile import ImagejRoi
import attiicc as ac
from attiicc import segment_anything as sa

class SamSegmenter:
    '''Segment nanowell images using a pretrained SAM model.
    
    This interface is designed to be used with nanowell images. 
    '''

    def __init__( 
        self,
        model_path: str = None,
        model_type: str = "vit_h",
        image_path: str = None,
    ) -> None:
        '''
        Initialize a SAM model and calculate the segmentation for an image.
        Inputs:
            model_path (str): The path to the model checkpoint. This must be downloaded
                from Meta on a user's local machine. Checkpoints can be downloaded from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
            model_type (str, optional): Specify the sam model type to load.
            Default is "vit_h". Can use "vit_b", "vit_l", or "vit_h".
            image_path (str, optional): The path to the image to segment. 
        Outputs:
            None
        '''
        assert isinstance(model_path, str), "Model checkpoint path on local machine must be specified. \
            Model checkpoints must be downloaded from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints."
        self.model_path = model_path
        self.model_type = model_type
        self._image_path = image_path
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

        if image_path is not None:
            self.sam = self._load_sam_model(model_type=model_type)
            self._sam_result, self.img_bgr = self._segment_image(self.sam, self._image_path)
        if self._sam_result is not None:
            self.segmentation = [mask["segmentation"] for mask in self._sam_result]
            self.area = [mask["area"] for mask in self._sam_result]
            self.bbox = [mask["bbox"] for mask in self._sam_result]
            self.predicted_iou = [mask["predicted_iou"] for mask in self._sam_result]
            self.point_coords = [mask["point_coords"] for mask in self._sam_result]
            self.stability_score = [mask["stability_score"] for mask in self._sam_result]
            self.crop_box = [mask["crop_box"] for mask in self._sam_result]
    
    @property
    def image_path(self):
        return self._image_path
    
    @image_path.setter
    def image_path(self, new_image_path):
        '''
        Update the image path and recalculate the segmentation results 
        without re-loading the SAM model.
        '''
        self._image_path = new_image_path
        self._sam_result, self.img_bgr = self._segment_image(self.sam, self._image_path)
        self.segmentation = [mask["segmentation"] for mask in self._sam_result]
        self.area = [mask["area"] for mask in self._sam_result]
        self.bbox = [mask["bbox"] for mask in self._sam_result]
        self.predicted_iou = [mask["predicted_iou"] for mask in self._sam_result]
        self.point_coords = [mask["point_coords"] for mask in self._sam_result]
        self.stability_score = [mask["stability_score"] for mask in self._sam_result]
        self.crop_box = [mask["crop_box"] for mask in self._sam_result]

    @property
    def sam_result(self):
        return self._sam_result
    
    @sam_result.setter
    def sam_result(self, target_area=[11500,13600]):
        '''
        Filter the SAM results by area.
        Inputs:
            target_area (list, optional): The target area of the ROI.
                Default is [11500,13600] which is the area of the nanowells where 
                the first value is the lower bound and the second value is the upper bound.
        '''
        self._sam_result = [mask for mask in self._sam_result if target_area[0] < mask['area'] < target_area[1]]
        self.segmentation = [mask["segmentation"] for mask in self._sam_result]
        self.area = [mask["area"] for mask in self._sam_result]
        self.bbox = [mask["bbox"] for mask in self._sam_result]
        self.predicted_iou = [mask["predicted_iou"] for mask in self._sam_result]
        self.point_coords = [mask["point_coords"] for mask in self._sam_result]
        self.stability_score = [mask["stability_score"] for mask in self._sam_result]
        self.crop_box = [mask["crop_box"] for mask in self._sam_result]

    def _load_sam_model(self, model_type) -> sa.Sam:
        '''
        Loads a pretrained SAM model.
        Input:
            model_type (str): Specify the sam model type to load.
            Can use "vit_b", "vit_l", or "vit_h". 
        Output:
            Sam: The loaded SAM model.
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


    def _segment_image(self, sam, image_path) -> Tuple[Dict, np.ndarray]:
        '''
        Segments an image using a pretrained SAM model.
        Inputs:
            sam (Sam): The loaded SAM model.
            image_path (str): The path to the image to segment.
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
        image_bgr = cv2.imread(image_path) # cv2 reads in BGR format
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # convert to RGB
        sam_result = mask_generator.generate(image_rgb)
        return sam_result, image_bgr
    
    def plot_segmented_image(self, titles=['Source Image', 'Segmented Image'], 
                             save=False, save_path=None) -> None:
        '''
        Plots the original image and the segmented image side-by-side.
        Inputs:
            titles (list, optional): The titles to use for the images. Default is ['Source Image', 'Segmented Image'].
            save (bool, optional): Whether to save the image. Default is False.
            save_path (str, optional): The path to save the image. Default is None.
        Outputs:
            None
        '''
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=self._sam_result)
        annotated_image = mask_annotator.annotate(scene=self.img_bgr.copy(), detections=detections)
        # Create a figure and a 1x2 grid of axes
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the images onto the axes
        axs[0].imshow(self.img_bgr)
        axs[0].set_title('source image')
        axs[0].axis('off')

        axs[1].imshow(annotated_image)
        axs[1].set_title('segmented image')
        axs[1].axis('off')

        if save:
        # Save the figure
            fig.savefig(save_path)
        # Display the figure
        plt.show()
        return 

    def plot_masks(self, save=False, save_path=None, 
                   grid_size=(30, 5), size=(48,96)) -> None:
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

    def generate_rois(self, target_area=[11500,13600],
                            similarity_filter=10,
                            roi_path=None,
                            roi_archive=True,
                            validation_plot=False,
                            validation_path=None):
        '''
        Generate ROIs from the segmentation results.
        Inputs:
            target_area (int, optional): The target area of the ROI. Default is 11500.
            similarity_filter (int, optional): When filtering for duplicate ROIs
                this method will search for ROI centroids that are within +/-
                a number of pixels (similarity_filter). Default is 10 pixels.
            roi_path (str, optional): The path to a directory where ROIs can be saved. Default is None.    
            roi_archive (bool, optional): Whether to save the ROIs as a .zip. Default is True.
                This will save the roi.zip file in the roi_path. Default is True.        
        Outputs:
            roi_dict (dict): A dictionary containing the roi number sorted by 
             y-axis centroid value as the key (str) and its corresponding roi object
             as the value.
        '''
        image_name = os.path.basename(self.image_path).rstrip(".png")
        seg_num = 1
        centroid_list = []
        duplicate_list = []
        roi_dict = {}
        self.sam_result = target_area
        for seg in self.segmentation:
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
                        coords_id = [cX, cY, roi, seg_num]
                        centroid_list.append(coords_id)
            seg_num += 1
        # Sort the list of lists by the value at index 0
        centroid_list_sorted = sorted(centroid_list, key=lambda x: x[0])
        # Remove duplicates
        for i in range(len(centroid_list_sorted) - 1):
            x, y, roi, seg_num = centroid_list_sorted[i]
            x_next, y_next, roi, seg_num = centroid_list_sorted[i + 1]
            if abs(x - x_next) < similarity_filter and abs(y - y_next) < similarity_filter:
                duplicate_list.append(centroid_list_sorted[i])
                print("Duplicate Value 1:", (x, y), "Duplicate Value 2:", (x_next, y_next))
        centroid_list_sorted = [x for x in centroid_list_sorted if x not in duplicate_list]
        print("Total number of ROIs: ", len(centroid_list_sorted))
        y_centroid_list_sorted = sorted(centroid_list_sorted, key=lambda x: x[1])
        for i in range(len(y_centroid_list_sorted)):
            roi_dict[i] = y_centroid_list_sorted[2]
        if roi_path is not None:
            print("Saving ROIs to: ", roi_path+'/'+image_name)
            new_path = os.path.join(roi_path, image_name)
            if not os.path.exists(new_path):
                print("Making directory at: ", new_path)
                os.makedirs(new_path)
            for i, j in enumerate(centroid_list_sorted):
                roi = j[2]
                roi_name = f"{image_name}_ROI_{i+1}.roi"
                roi.tofile(os.path.join(new_path, roi_name))
            print(f"ROIs saved for {image_name}")
            if roi_archive:
                print("Archiving ROIs to: ", f'{roi_path}/{image_name}_roi.zip')
                with zipfile.ZipFile(f'{roi_path}/{image_name}_rois.zip', 'w') as zipf:
                    for root, _, files in os.walk(new_path):
                        for file in files:
                            zipf.write(os.path.join(root, file), file)
                print(f"ROIs archived for {image_name}")
            if validation_plot:
                centroids = [(x, y) for x, y, roi, seg_num in y_centroid_list_sorted]
                seg_num = [seg_num for x, y, roi, seg_num in y_centroid_list_sorted]
                # Create a scatter plot of the centroids
                plt.scatter(*zip(*centroids), marker='o')
                plt.title(f"Centroids for {image_name}")
                # Annotate each point with its label
                for (x, y), i in zip(centroids, range(len(centroids))):
                    plt.text(x, y, str(i))
                if validation_path is None:
                    validation_dir = os.path.join(roi_path, "validation_plots")
                    if not os.path.exists(validation_dir):
                        print("Making directory at: ", validation_dir)
                        os.makedirs(validation_dir)
                    plt.savefig(os.path.join(validation_dir, f"{image_name}_validation.png"))
                else:
                    plt.savefig(os.path.join(validation_path, f"{image_name}_validation.png"))
        return roi_dict 
