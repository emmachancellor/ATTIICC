import torch
import cv2
import os
import segment_anything
from typing import Dict, Tuple, List

from .segmentation import Segmentation


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

    def segment(self, png_path: str) -> Segmentation:
        """Segment an image using a pretrained SAM model.

        Args:
            sam (Sam): The loaded SAM model.
            png_path (str): The path to the image to segment.

        Returns:
            sam_result (Segmentation): The segmentation object.

        """
        # Load the image.
        image_bgr = cv2.imread(png_path) # cv2 reads in BGR format
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # convert to RGB

        # Segment the image.
        mask_generator = segment_anything.SamAutomaticMaskGenerator(self.sam)
        sam_result = mask_generator.generate(image_rgb)

        # Build the Segmentation object.
        sam_segmentation = Segmentation(sam_result, image_bgr, png_path=png_path)

        return sam_segmentation
