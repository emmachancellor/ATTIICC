import numpy as np
import torch
import cv2
import os
import supervision as sv
import matplotlib.pyplot as plt
from typing import Dict, Tuple
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
        self.image_path = image_path
        self.sam_result = None
        self.img_bgr = None
        self.sam = None
        self.sam_result = None
        self.segmentation = None
        self.area = None
        self.bbox = None
        self.predicted_iou = None
        self.point_coords = None
        self.stability_score = None
        self.crop_box = None

        if image_path is not None:
            self.sam = self._load_sam_model(model_type=model_type)
            self.sam_result, self.img_bgr = self._segment_image(self.sam, self.image_path)
        if self.sam_result is not None:
            self.segmentation = [mask["segmentation"] for mask in self.sam_result]
            self.area = [mask["area"] for mask in self.sam_result]
            self.bbox = [mask["bbox"] for mask in self.sam_result]
            self.predicted_iou = [mask["predicted_iou"] for mask in self.sam_result]
            self.point_coords = [mask["point_coords"] for mask in self.sam_result]
            self.stability_score = [mask["stability_score"] for mask in self.sam_result]
            self.crop_box = [mask["crop_box"] for mask in self.sam_result]
    
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
        detections = sv.Detections.from_sam(sam_result=self.sam_result)
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
        in sorted(self.sam_result, key=lambda x: x['area'], reverse=True)]

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
