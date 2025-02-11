import torch
import cv2
import os
import numpy as np
import segment_anything
from typing import List, Optional

from .segmentation import Segmentation, Plate, GridDefinition
from . import utils


# -----------------------------------------------------------------------------

class SamSegmenter:
    """Segment nanowell images using a pretrained SAM model.

    This interface is designed to be used with nanowell images.
    """

    def __init__(self, weights: str = None, model_type: str = "vit_h") -> None:
        """
        Initialize a SAM model and calculate the segmentation for an image.

        Args:
            model_type (str, optional): Specify the sam model type to load.
                Default is "vit_h". Can use "vit_b", "vit_l", or "vit_h".
            weights (str): The path to the model checkpoint. This must be downloaded
                from Meta on a user's local machine. Checkpoints can be downloaded from
                https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

        """
        self._validate_weights(weights)
        self.model_type = model_type
        self.weights = weights

        # Load model
        self.sam = self._load_sam_model(model_type=model_type)

    # --- Internal ---------------------------------------------------------------

    def _validate_weights(self, weights: str) -> None:
        """Validate the model weights path."""

        if not isinstance(weights, str):
            raise ValueError(
                "Model checkpoint path on local machine must be specified for segmentation. "
                "Model checkpoints may be downloaded from "
                "https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints."
            )

    def _load_sam_model(
        self,
        model_type: str
    ) -> "segment_anything.Sam":
        """Loads a pretrained SAM model.

        Args:
            model_type (str): Specify the sam model type to load.
                Can use "vit_b", "vit_l", or "vit_h".

        Returns:
            sam: The loaded SAM model.

        """
        if torch.cuda.is_available():
            print("CUDA is available.")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        else:
            print("CUDA is not available.")

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = model_type

        # Load the model
        sam = segment_anything.sam_model_registry[MODEL_TYPE](checkpoint=self.weights).to(device=DEVICE)
        print("Model Loaded")
        return sam

    # --- Public ---------------------------------------------------------------

    def segment(self, 
                image_path: str,
                use_og_img: bool = False) -> Segmentation:
        """Segment an image using a pretrained SAM model.

        Args:
            sam (Sam): The loaded SAM model.
            image_path (str): The path to the image to segment.

        Returns:
            sam_result (Segmentation): The segmentation object.

        """
        # Load the image.
        if utils.is_tif(image_path):
            image_rgb = utils.load_tif(image_path).convert('RGB')
            image_rgb = np.asarray(image_rgb)
        else:
            image_bgr = cv2.imread(image_path) # cv2 reads in BGR format
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # convert to RGB
        # Load 16-bit version of image for later use
        if utils.is_tif(image_path):
            og_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        else:
            og_img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        # Segment the image.
        mask_generator = segment_anything.SamAutomaticMaskGenerator(self.sam)
        sam_result = mask_generator.generate(image_rgb)

        # Build the Segmentation object.
        if use_og_img:
            sam_segmentation = Segmentation(sam_result, og_img, image_path=image_path)
        else:
            sam_segmentation = Segmentation(sam_result, image_rgb, image_path=image_path)

        return sam_segmentation

    def build_plate(self, 
                    image, 
                    grid_definition: Optional[GridDefinition] = None,
                    use_og_img: bool = False) -> Plate:
        """Build a Plate object from a grid definition.

        Args:
            grid_definition (GridDefinition): The grid definition object.

        Returns:
            plate (Plate): The plate object.

        """
        # Segment the image.
        segmentation = self.segment(image, use_og_img)

        # Find the wells.
        plate = segmentation.find_wells()

        # Apply the grid definition, if supplied.
        if grid_definition is not None:
            plate = plate.apply_grid(grid_definition)

        return plate

    def build_plates(self, 
                     images: List[str], 
                     grid_definition: Optional[GridDefinition] = None,
                     use_og_img: bool = False) -> List[Plate]:
        """Build a list of Plate objects from a list of image paths.

        These can then be stacked with a PlateStack object.

        Args:
            images (List[str]): The list of image paths.
            grid_definition (GridDefinition): The grid definition object.

        Returns:
            plates (List[Plate]): The list of plate objects.

        """
        plates = [self.build_plate(image, grid_definition, use_og_img) for image in images]
        return plates
