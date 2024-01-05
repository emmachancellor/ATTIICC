import numpy as np
import torch
import cv2
import os
import supervision as sv
from attiicc import segment_anything as sa

def load_sam_model(model_type: str) -> sa.Sam:
    '''
    Loads a pretrained SAM model.
    Input:
        model_type (str): Specify the sam model type to load.
        Can use "vit-b", "vit-l", or "vit-h". 
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
    HOME = os.getcwd()
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    MODEL_TYPE = model_type
    # Load the model
    sam = sa.sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    print("Model Loaded")
    return sam

def segment_image(sam: sa.Sam, img_dir_path: str):
    '''
    Segments an image using a pretrained SAM model.
    Inputs:
        sam (Sam): The pretrained SAM model.
        img_dir_path (str): The path to the image to segment.
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
    image_bgr = cv2.imread(img_dir_path) # cv2 reads in BGR format
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # convert to RGB
    sam_result = mask_generator.generate(image_rgb)
    return sam_result