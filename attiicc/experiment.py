import os
import numpy as np
import fnmatch
import re
import cv2
import imagej
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from .segmentation import SamSegmenter
from .utils import convert_tif_to_png, generate_comparison_plot, sort_paths


class NanoExperiment(SamSegmenter):
    '''Apply segmentation functions to an experiment with multiple channels, 
    fields of view, and time points.

    This interface is designed to be used with many directories of
    nanowell images, where the end goal is to create directories
    for each individual well containing images from every time point collected. 
    
    The final number of segmented images equates to:
    (number of channels) x (number of fields of view) x (number of wells per field) x (number of time points)
    '''

    def __init__(
        self,
        experiment_path: str = None,
        field_id: str = None,
        num_fields: int = None,
        channel_id: str = None,
        num_channels: int = None,
        time_point_id: str = None,
        num_time_points: int = None,
        segment_channel: int = None,
        field_leading_zero: bool = False,
        time_point_leading_zero: bool = False,
        well_dict: dict = None,
        whole_image_dict: dict = None,
        **kwargs # For SamSegmenter
    ) -> None:
        '''Initialize a NanoExperiment object.
        The experimental data should be organized as a directory tree with the following structure:
        experiment_path/
            field_id_0_channel_id_0/
                image_time_point_id_0.tif
                image_time_point_id_1.tif
                ...
            field_id_0_channel_id_1/
                image_time_point_id_0.tif
                image_time_point_id_1.tif
                ...
            ...
        The experiment directory should have subdirectories that are named according
        to the field of view and channel identifiers. Each subdirectory should contain
        the raw .TIF images, with the time point identifier in the file name.
        
        Args:
            experiment_path: (str) The path to the experiment parent directory.
            field_id: (str) The field of view identifier used in direcotry names.
            num_fields: (int) The number of fields of view in the experiment.
            channel_id: (str) The channel identifier used in directory names.
            num_channels: (int) The number of channels in the experiment.
            time_point_id: (str) The time point identifier used in file names.
            num_time_points: (int) The number of time points in the experiment.
            segment_channel: (int) The channel to use for segmentation. Default is None.
            field_leading_zero: (bool) Whether the field of view identifier has a leading zero.
            **kwargs: Keyword arguments for SamSegmenter by method. 
                generate_rois_params: (dict) Keyword arguments for SamSegmenter.generate_rois().
                plot_segmented_image_params: (dict) Keyword arguments for SamSegmenter.plot_segmented_image().
                plot_masks_params: (dict) Keyword arguments for SamSegmenter.plot_masks().
                For all parameters, format should be: {'parameter_name': parameter_value}
                Example: generate_rois_params = {'save_rois': True, 'save_directory': 'experiment_path/ROI'}
        '''
        assert isinstance(experiment_path, str), 'Experiment_path must be a string'
        assert isinstance(field_id, str), 'field_id must be a string specifying the field of view identifier, Ex. \'f\''
        assert isinstance(num_fields, int), 'num_fields must be an integer specifying the number of fields of view'
        assert isinstance(channel_id, str), 'channel_id must be a string specifying the channel identifier, Ex. \'c\''
        assert isinstance(num_channels, int), 'num_channels must be an integer specifying the number of channels'
        assert isinstance(time_point_id, str), 'time_point_id must be a string specifying the time point identifier, Ex. \'t\''
        assert isinstance(num_time_points, int), 'num_time_points must be an integer specifying the number of time points'
        sam_kwargs = {key: kwargs[key] for key in kwargs if key not in ['generate_rois_params'] 
                      and key not in ['plot_segmented_image_params'] and key not in ['plot_masks_params']}
        self._sam_kwargs = sam_kwargs
        super().__init__(**self._sam_kwargs)
        self._experiment_path = experiment_path
        self._field_id = field_id
        self._num_fields = num_fields
        self._channel_id = channel_id
        self._num_channels = num_channels
        self._time_point_id = time_point_id
        self._segment_channel = segment_channel
        self._num_time_points = num_time_points
        self._field_leading_zero = field_leading_zero
        self._time_point_leading_zero = time_point_leading_zero
        self.well_dict = well_dict
        self.whole_image_dict = whole_image_dict
        self.structure = f'Experiment Structure \n\
            segment_channel: {self._segment_channel} \n\
            field_id: {self._field_id} \n\
            field_num: {self._num_fields} \n\
            field_leading_zero: {self._field_leading_zero} \n\
            channel_id: {self._channel_id} \n\
            channel_num: {self._num_channels} \n\
            time_point_id: {self._time_point_id} \n\
            time_point_num: {self._num_time_points} \n\
            time_point_leading_zero: {self._time_point_leading_zero}'
        self.generate_rois_params = kwargs.get('generate_rois_params', {})
        self.plot_segmented_image_params = kwargs.get('plot_segmented_image_params', {})
        self.plot_masks_params = kwargs.get('plot_masks_params', {})
    
    @property
    def sam_kwargs(self) -> dict:
        return self._sam_kwargs

    @sam_kwargs.setter
    def sam_kwargs(self, sam_kwargs) -> None:
        self._sam_kwargs = sam_kwargs

    @property
    def field_leading_zero(self) -> bool:
        return self._field_leading_zero
    
    @field_leading_zero.setter
    def field_leading_zero(self, field_leading_zero) -> None:
        self._field_leading_zero = field_leading_zero

    @property
    def time_point_leading_zero(self) -> bool:
        return self._time_point_leading_zero
    
    @time_point_leading_zero.setter
    def time_point_leading_zero(self, time_point_leading_zero) -> None:
        self._time_point_leading_zero = time_point_leading_zero
    
    def set_structure(
        self,
        field_id: str,
        num_fields: int,
        channel_id: str,
        num_channels: int,
        time_point_id: str,
        num_time_points: int,
        segment_channel: int,
        field_leading_zero: bool,
        time_point_leading_zero: bool
    ) -> None:
        """
        Set the structure of the experiment.

        Args:
            field_id: (str) The field of view identifier used in direcotry names.
            num_fields: (int) The number of fields of view in the experiment.
            channel_id: (str) The channel identifier used in directory names.
            num_channels: (int) The number of channels in the experiment.
            time_point_id: (str) The time point identifier used in file names.
            num_time_points: (int) The number of time points in the experiment.
            segment_channel: (int) The channel to use for segmentation. Default is None.
            field_leading_zero: (bool) Whether the field of view identifier has a leading zero.
            time_point_leading_zero: (bool) Whether the time point identifier has a leading zero.
            **kwargs: Keyword arguments for SamSegmenter by method. 
                generate_rois_params: (dict) Keyword arguments for SamSegmenter.generate_rois().
                plot_segmented_image_params: (dict) Keyword arguments for SamSegmenter.plot_segmented_image().
                plot_masks_params: (dict) Keyword arguments for SamSegmenter.plot_masks().
                For all parameters, format should be: {'parameter_name': parameter_value}
                Example: generate_rois_params = {'save_rois': True, 'save_directory': 'experiment_path/ROI'}

        Returns:
            None

        """
        self._field_id = field_id
        self._num_fields = num_fields
        self._channel_id = channel_id
        self._num_channels = num_channels
        self._time_point_id = time_point_id
        self._num_time_points = num_time_points
        self._segment_channel = segment_channel
        self._field_leading_zero = field_leading_zero
        self._time_point_leading_zero = time_point_leading_zero
        return print(self.structure)

    def generate_image_dicts(
        self,
        total_rois: int,
        field_str: str,
        png_path: str,
        roi: list,
        box: list,
        centroids: list,
        whole_image_dict = None,
        well_dict = None,
        first_frame = False,
        img_idx = None,
        well_location_tolerance = 15
    ) -> dict:
        """
        Update a dictionary containing well information across each time point \
        for each field and update the whole_image_dict with the sum of the bounding boxes
        of the first well in each time point.

        Args:
            total_rois: (int) The total number of ROIs segmented in the image.
            field_str: (str) The field of view identifier. Derived from self._field_id.
            png_path: (str) The path to the image in the time series.
            roi: (list) A list of the ROIs segmented in the image.
            box: (list) A list of the bounding box coordinates for each ROI.
            centroids: (list) A list of the centroids for each well ROI.
            whole_image_dict: (dict) A dictionary containing the field level information 
                for each nanowell image.
            well_dict: (dict) A dictionary containing the ROIs, bounding box coordinates,
                and time point identifier for each well across all time points. Default is None.
                If left as None, the function will initialize an empty dictionary.
            img_idx: (int) The index of the image in the time series. Default is None.
            well_location_tolerance: (int) The allowed difference in the centroid of the bounding box
                of a given well and the bounding box of the same well in the previous time point.
                Default is 5 pixels.
        
        Returns:
            well_dict (dict): A dictionary containing the ROIs, bounding box coordinates,
                and time point identifier for each well across all time points. 
                The dictionary is structured as follows:
                {'field_id_0_well_id_0': [[roi_0, roi_1, ...], 
                                        [box_0, box_1, ...], 
                                        [time_point_0, time_point_1, ...],
                                        [num_rois_0, num_rois_1, ...]],
                'field_id_0_well_id_1': [[roi_0, roi_1, ...],
                                        [box_0, box_1, ...],
                                        [time_point_0, time_point_1, ...],
                                        [num_rois_0, num_rois_1, ...]]}
            whole_image_dict (dict): The updated dictionary containing the field level information for each time point.
                The dictionary is structured as follows:
                {'field_id_0': [[total_rois], {png_path: {'well_id_0': (centroid_x, centroid_y)}, ...],
                'field_id_1': [[total_rois], {png_path: {'well_id_0': (centroid_x, centroid_y)}, ...]}

        """
        # Create new image-level dictionary to keep track of this image's wells
        whole_field_wells = {}
        if well_dict is None:
            well_dict = {}        
        well_number = 0
        print("Matching ROIS for: ", png_path)
        for result_index in range(len(roi)):
            total_rois = len(roi)
            well_name = f'{field_str}_well{well_number}'
            png_name = png_path.split('/')[-1]
            if well_name not in well_dict or first_frame:
                well_dict[well_name] = [[roi[result_index]], [box[result_index]], [png_name], [centroids[result_index]]]
                whole_field_wells[well_name] = centroids[result_index]
                well_number += 1
            else:
                x_1, y_1 = well_dict[well_name][3][-1]
                last_timepoint_location_sum = x_1 + y_1
                x_2, y_2 = centroids[result_index]
                current_timepoint_location_sum = x_2 + y_2
                #TODO: Fix the logic of finding the matching centroid
                if abs(current_timepoint_location_sum - last_timepoint_location_sum) > well_location_tolerance:
                    print(f'Well {well_name} has moved more than {well_location_tolerance} pixels from the previous time point.')
                    print(f'Last time point location: {x_1, y_1}')
                    print(f'Current time point location: {x_2, y_2}')
                    # Search for a matching centroid from all centroids within the timpoint
                    matching_centroid_index = None
                    for i, (x, y) in enumerate(centroids):
                        #if abs((x + y) - last_timepoint_location_sum) <= well_location_tolerance:
                        if abs(x - x_1) <= well_location_tolerance:
                            matching_centroid_index = i
                            break
                    
                    if matching_centroid_index is not None:
                        print(f"Found matching centroid at index {matching_centroid_index}, (time point location: {(x,y)})")
                        well_dict[well_name] = [well_dict[well_name][0] + [roi[matching_centroid_index]], 
                                                well_dict[well_name][1] + [box[matching_centroid_index]],
                                                well_dict[well_name][2] + [png_name], 
                                                well_dict[well_name][3] + [centroids[matching_centroid_index]]]
                        well_number += 1
                        ### CHECK THIS LINE
                        whole_field_wells[well_name] = centroids[matching_centroid_index]
                        print(f"Added ROI to {whole_field_wells[well_name]}")
                        continue
                    else:
                        print(f"No matching centroid found for {well_name}. No ROI will be added at this time point.")
                        well_number += 1
                        whole_field_wells[well_name] = "No Matching Well"
                        continue
                
                else: # If the centroid hasn't moved significantly, update the existing well
                    well_dict[well_name] = [well_dict[well_name][0] + [roi[result_index]], 
                                            well_dict[well_name][1] + [box[result_index]],
                                            well_dict[well_name][2] + [png_name], 
                                            well_dict[well_name][3] + [centroids[result_index]]]
                    whole_field_wells[well_name] = centroids[result_index]
                    well_number += 1
        if whole_image_dict is None:
            whole_image_dict = {}
        if field_str not in whole_image_dict.keys():
            # Create initial field-level dictionary entry
            print("Adding field to whole_image_dict: ", field_str)
            whole_image_dict[field_str] = [[total_rois], [{png_path: whole_field_wells}]]
        else:
            # Update field-level dictionary entry with new image information
            whole_image_dict[field_str][1][0][png_path] = whole_field_wells
            print("Updating field with png-level information in whole_image_dict: ", png_path)
            whole_image_dict[field_str] = [whole_image_dict[field_str][0] + [total_rois],
                                            whole_image_dict[field_str][1]] 
        return well_dict, whole_image_dict

    def generate_validation_plot(
        self,
        whole_image_dict: dict,
        validation_path: str,
        roi_path: str
    ) -> None:
        """Generate a plot with well labels on the segmented image.

        Args:
            whole_image_dict: (dict) A dictionary containing the ROI coordinates and 
                image information for each well across all time points.
                The dictionary is structured as follows:
                {'field_id_0': [[total_rois], {png_path: {'well_id_0': (centroid_x, centroid_y)}, 
                {png_path:(centroid_x, centroid_y)}...],
                'field_id_1': [[total_rois], {png_path: {'well_id_0': (centroid_x, centroid_y)}, 
                {png_path:(centroid_x, centroid_y)}...]}

        """
        for field in whole_image_dict.keys():
            time_point_dict = whole_image_dict[field][1][0]
            image_file_paths = list(time_point_dict.keys())
            image_file_paths = sort_paths(image_file_paths)
            for png_path in image_file_paths:
                img = mpimg.imread(png_path)
                image_name = png_path.split('/')[-1].rstrip('.png')
                plt.imshow(img, cmap='gray')
                print("Generating validation plot for ", image_name)
                time_point_dict = whole_image_dict[field][1]
                well_coord_dict = time_point_dict[0][png_path]
                print("Well coordinates: ", well_coord_dict)
                # Extract coordinates and labels from the dictionary
                coordinates = list(well_coord_dict.values())
                labels = list(well_coord_dict.keys())

                # Create a scatter plot of the centroids
                print("Coordinates: ", coordinates)
                plt.scatter(*zip(*coordinates), color='yellow', marker='o')
                plt.title(f"Centroids for {image_name}")

                # Annotate each point with its well name
                for (x, y), label in zip(coordinates, labels):
                    strip_label = label.split('_')[-1]
                    plt.text(x, y, strip_label, color='white')
                if validation_path is None:
                    validation_dir = os.path.join(roi_path, "validation_plots")
                    if not os.path.exists(validation_dir):
                        print("Making directory at: ", validation_dir)
                        os.makedirs(validation_dir)
                    plt.savefig(os.path.join(validation_dir, f"{image_name}_validation.png"))
                elif not os.path.exists(validation_path):
                    os.makedirs(validation_path)
                    plt.savefig(os.path.join(validation_path, f"{image_name}_validation.png"))
                else:
                    plt.savefig(os.path.join(validation_path, f"{image_name}_validation.png"))        
                plt.close()
        

    def segment_nanowells(
        self,
        model_path: str = None,
        model_type: str = 'vit_h',
        output_directory: str = None,
        convert_png: bool = False,
        **kwargs
    ) -> dict:
        """Segment and crop the images in the experiment.

        Takes in images that are ordered by the experiment structure, then
        segments and crops the images well-wise. The cropped images and their
        ROIs will be saved in individual directories for each well. The structure
        of the output directory will be:
            output_directory/
                ROI/
                    field_id_0_channel_id_0_well_id_0/
                        image_time_point_id_0_well_id_0.roi
                        image_time_point_id_1_well_id_0.roi
                        ...

        The path to tif images must be named as /field_idfield_numchannel_idchannel_num
        
        Example: 
            field_id = 'f'
            field_num = 0
            channel_id = 'd'
            channel_num = 3
            field_leading_zero = True
            Path to TIF images: experiment_path/f00d3

        Note: if the field_id has a leading zero, the 'field_leading_zero' parameter in the SamSegmenter 
            instance must be set to True.

        Args:
            model_path: (str) The path to the model checkpoint. This must be downloaded \
                from Meta on a user's local machine. Checkpoints can be downloaded from \
                https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
            model_type (str, optional): Specify the sam model type to load. \
                Default is "vit_h". Can use "vit_b", "vit_l", or "vit_h".
            output_directory: (str) The path to the output directory. This will house \
                subdirectories for the ROIs and cropped images.
            convert_png: (bool) If True, the TIF images will be converted to PNG and saved in a new directory. \
                These images will be used for segmentation. If PNG images exist, \
                the directory must follow the same structure as the TIF directory, with \
                the directory extension 'png.' For example, '/experiment_path/field_id_0_channel_id_0_png/'. \
                Default is False.
            
        Returns:
            well_dict (dict): A dictionary containing the ROIs, bounding box coordinates,
                and time point identifier for each well across all time points. 
                The dictionary is structured as follows:
                {'field_id_0_well_id_0': [[roi_0, roi_1, ...], 
                                        [box_0, box_1, ...], 
                                        [time_point_0, time_point_1, ...],
                                        [num_rois_0, num_rois_1, ...]],
                'field_id_0_well_id_1': [[roi_0, roi_1, ...],
                                        [box_0, box_1, ...],
                                        [time_point_0, time_point_1, ...],
                                        [num_rois_0, num_rois_1, ...]]}
            whole_image_dict: (dict) A dictionary containing the ROI coordinates and 
                image information for each well across all time points.
                The dictionary is structured as follows:
                {'field_id_0': [[total_rois], {png_path: {'well_id_0': (centroid_x, centroid_y)}, ...],
                'field_id_1': [[total_rois], {png_path: {'well_id_0': (centroid_x, centroid_y)}, ...]}

        """
        assert isinstance(model_path, str), "Model checkpoint path on local machine must be specified for segmentation. \
            Model checkpoints must be downloaded from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints."
        self.generate_rois_params = kwargs.get('generate_rois_params', {})
        self.plot_segmented_image_params = kwargs.get('plot_segmented_image_params', {})
        self.plot_masks_params = kwargs.get('plot_masks_params', {})
        # Extract well_location_tolerance to specify the allowed difference in the centroid of the bounding box
        # between well locations in adjacent time points. Default is 5 pixels.
        well_location_tolerance = kwargs.get('well_location_tolerance', 15)
        if output_directory is None:
            roi_path = self._experiment_path + '/ROI'
        else: 
            roi_path = output_directory + '/ROI'
        if not os.path.exists(roi_path):
            os.makedirs(roi_path)
        # Get path to image for segmentation
        channel_str = self._channel_id + str(self._segment_channel)
        # Locate and/or create PNG images needed for segmentation from brightfield images
        whole_image_dict = None
        well_dict = None
        for f in range(self._num_fields):
            if self._field_leading_zero and f < 10:
                field_str = self._field_id + '0' + str(f)
            else:
                field_str = self._field_id + str(f)
            tif_image_directory_path = f'{self._experiment_path}/{field_str}{channel_str}'
            if convert_png:
                png_image_directory_path = convert_tif_to_png(tif_image_directory_path)
            else:
                png_image_directory_path = tif_image_directory_path + '_png'
            if not os.path.exists(png_image_directory_path):
                raise ValueError(f'No .png images found in {png_image_directory_path}. \
                    Please convert the .TIF images to .png using the convert_png parameter.')
            # With PNG images, segment nanowells in images
            begin_segmenting = True
            # Segment images at the single image level and iteratively update the
            # whole_image_dict and well_dict for each time point
            # PNG PATHS MUST BE SORTED CHRONOLOGICALLY!!!
            sorted_png_image_list = sort_paths(os.listdir(png_image_directory_path))
            for i, j in enumerate(sorted_png_image_list):
                first_frame = (i == 0)
                png_path=png_image_directory_path + '/' + j
                tif_path=tif_image_directory_path + '/' + j.rstrip('.png') + '.TIF'
                if begin_segmenting is True: # Initialize SamSegmenter instance
                    segmentation = SamSegmenter(weights=model_path,
                                                    model_type=model_type, 
                                                    png_path=png_path,
                                                    tif_path=tif_path)
                    begin_segmenting = False
                else: # If SamSegmenter instance already exists, update the image path, don't need to re-load SAM model
                    segmentation.update_image(png_path, tif_path)
                roi, box, centroids = segmentation.generate_rois(**self.generate_rois_params)
                well_dict, whole_image_dict = self.generate_image_dicts(len(roi), 
                                                                      field_str, 
                                                                      png_path, 
                                                                      roi, 
                                                                      box,
                                                                      centroids, 
                                                                      whole_image_dict,
                                                                      well_dict,
                                                                      first_frame=first_frame,
                                                                      img_idx=i,
                                                                      well_location_tolerance=well_location_tolerance)
        # Generate validation plots, if specified
        if self.generate_rois_params.get('validation_plot') is True:
            self.generate_validation_plot(whole_image_dict, 
                                          self.generate_rois_params.get('validation_path'),
                                          roi_path)
        return whole_image_dict, well_dict

    
    