# ATTIICC 🪜🔦
## Automated Temporal Tumor Immune Interaction Chamber Capture

### Overview
This pipeline was developed to automate image processing and analysis of time-based tummor-immune interactions captured by fluoresence microscopy in nanowell chambers. The pipeline has the following main processes:
1. **[Image Processing](#image-processing)**
    - **Pre-processing**: Image format conversion, if needed, for downstream analysis (typically .TIF to .PNG to perform image segmentation)
    - **Nanowell segmentation**: Identifying all nanowells in a single image and capturing each well's coordinates as a mask, or region of interest (ROI)
    - **Nanowell registration**: Mapping each well's ROI to the same well in images from other timepoints
    - Image cropping: Cropping a box around each well's ROI and saving cropped images as distinct files for downstream well-wise time-series analysis
2. Image Analysis
    - Cell segmentation: Identifying cells in each well's cropped image
    - Cell tracking: Tracking cells across timepoints to generate cell trajectories
    [TO COME]

### Image Processing
- Describe API [TO COME]
