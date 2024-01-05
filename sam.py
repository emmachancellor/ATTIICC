import numpy as np
import torch
import cv2
import os
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.modeling.sam import Sam

def load_sam_model(model_type: str) -> Sam:
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
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    print("Model Loaded")
    return sam

load_sam_model('vit_h')