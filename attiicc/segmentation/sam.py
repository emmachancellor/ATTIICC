import numpy as np
import torch
import cv2
import os
import supervision as sv
import pkg_resources
from attiicc import segment_anything as sa

class SamSegmenter:
    '''Segment nanowell images using a pretrained SAM model.
    
    This interface is designed to be used with nanowell images. 
    '''

    def __init__( 
        self,
        model_type: str = "vit_h",
        image_path: str = None,
    ) -> None:
        '''
        Initialize a SAM model and calculate the segmentation for an image.
        Inputs:
            model_type (str, optional): Specify the sam model type to load.
            Default is "vit_h". Can use "vit_b", "vit_l", or "vit_h".
            image_path (str, optional): The path to the image to segment. 
        Outputs:
            None
        '''
        self.model_type = model_type
        self.sam = self._load_sam_model(self.model_type)
        self.sam_result = None
        self.segmentation = None
        self.area = None
        self.bbox = None
        self.predicted_iou = None
        self.point_coords = None
        self.stability_score = None
        self.crop_box = None

        if image_path is not None:
            self.image_path = image_path
            self.sam_result = self._segment_image(self.sam, self.image_path)
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
        # Path to the checkpoint file
        if model_type == "vit_h":
            CHECKPOINT_PATH = pkg_resources.resource_filename('attiicc', 'segment_anything/weights/sam_vit_h_4b8939.pth')
        elif model_type == "vit_b":
            CHECKPOINT_PATH = pkg_resources.resource_filename('attiicc', 'segment_anything/weights/sam_vit_b_4b8939.pth')
        elif model_type == "vit_l":
            CHECKPOINT_PATH = pkg_resources.resource_filename('attiicc', 'segment_anything/weights/sam_vit_l_4b8939.pth')
        MODEL_TYPE = model_type
        # Load the model
        sam = sa.sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
        print("Model Loaded")
        return sam

    def _segment_image(self, sam, image_path):
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
        return sam_result