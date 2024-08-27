import os
import numpy as np
import fnmatch
import read_roi
import attiicc as ac
import cv2
import imagej
from attiicc.segmentation import SamSegmenter
from experiment_utils import convert_tif_to_png, find_files, generate_comparison_plot
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
        
        Inputs:
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
    
    def set_structure(self, 
                      field_id: str,
                      num_fields: int,
                      channel_id: str,
                      num_channels: int,
                      time_point_id: str,
                      num_time_points: int,
                      segment_channel: int,
                      field_leading_zero: bool,
                      time_point_leading_zero: bool) -> None:
        '''
        Set the structure of the experiment.

        Inputs:
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
        Outputs:
            None
        '''
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

    def generate_image_dicts(self,
                                        total_rois: int, 
                                        field_str: str,
                                        png_path: str,
                                        roi: list,
                                        box: list,
                                        centroids: list,
                                        whole_image_dict = None,
                                        well_dict = None,
                                        img_idx = None,
                                        well_location_tolerance = 5) -> dict:
        '''
        Updates a dictionary containing well information across each time point \
        for each field and update the whole_image_dict with the sum of the bounding boxes
        of the first well in each time point.

        Inputs:
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
        
        Outputs:
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
                {'field_id_0': {"total_rois": total_rois, 
                                "png_path": png_path, 
                                "first_well_coords": [sum_boxes]}
                'field_id_1': {"total_rois": total_rois,
                                "png_path": png_path,
                                "first_well_coords": [sum_boxes]}
        '''
        if well_dict is None:
            well_dict = {}        
        # Iterate over all ROIs, but must match each ROI to the correct well via pixel location
        well_number = 0
        for result_index in range(len(roi)):
            total_rois = len(roi)
            well_name = f'{field_str}_well{well_number}'
            potential_duplicate_well_name = f'{field_str}_well{well_number-1}'
            png_name = png_path.split('/')[-1]
            if well_name not in well_dict:
                well_dict[well_name] = [[roi[result_index]], [box[result_index]], [png_name], [centroids[result_index]]]
                print(f'Number of entries in well_dict: {len(well_dict)}')
                print('Index: ', result_index)
            else:
                print(f'Well {well_name} already exists in the dictionary.')
                Check if the centroid of the last entry is in the same location as the current ROI
                x_1, y_1 = well_dict[well_name][3][-1]
                print("Last time point centroid: ", x_1, y_1)
                last_timepoint_location_sum = x_1 + y_1
                x_2, y_2 = centroids[result_index]
                current_timepoint_location_sum = x_2 + y_2
                #print(f'Difference between last and current timepoint: {abs(current_timepoint_location_sum - last_timepoint_location_sum)}')
                if abs(current_timepoint_location_sum - last_timepoint_location_sum) > well_location_tolerance:
                    # Check if the mis-match is due to a duplicate well
                    #print('Potential Duplicate Name: ', potential_duplicate_well_name)
                    #print('Well Name: ', well_name)
                    if (well_number > 0) and (well_dict[well_name][3][-1] == well_dict[potential_duplicate_well_name][3][-1]):
                        print(f'Well {well_name} is a duplicate of {potential_duplicate_well_name}.')
                    # print(f'Well {well_name} has moved more than {well_location_tolerance} pixels from the previous time point.')
                    # print(f'Last time point location: {last_timepoint_location_sum}')
                    # print(f'Current time point location: {current_timepoint_location_sum}')
                    # print('Please verify the well location and update the ROI if necessary.')
                    # response = input(f"Do you want to add this well as a timepoint for {well_name}? (y/n): ")
                    # if response.lower() != "y":
                    #     continue
                    # if response.lower() == "y":
                    #     print("Adding well information to dictionary.")
                well_dict[well_name] = [well_dict[well_name][0] + [roi[result_index]], 
                                        well_dict[well_name][1] + [box[result_index]],
                                        well_dict[well_name][2] + [png_name], 
                                        well_dict[well_name][3] + [centroids[result_index]]]
            # Only increment the well number if a well is added to the dictionary
            # Don't want to increment the well if the ROI does not belong to a full well or duplicate well
            well_number += 1
        if whole_image_dict is None:
            whole_image_dict = {}
        if field_str not in whole_image_dict:
                        whole_image_dict[field_str] = {"total_rois": [total_rois],
                                "png_path": [png_path]}
        return well_dict, whole_image_dict

    def match_wells_v1(self,
                    whole_image_dict: dict,
                    well_dict: dict) -> None:
        '''
        Match wells across time points. If the well locations are not consistent across time points,\
        the user will be prompted to verify the discordance and given the opportunity to overwrite\
        the ROIs of the image that had an incorrect segmentation with the ROIs from the adjacent time point.\n
        Inputs:
            whole_image_dict: (dict) contains the field level information for the number of 
                ROIs in each image for each time point, the png path to each image in the time series, 
                and the sum of the bounding boxes for the ROI of the first well in each time point. 
                The sum of the bounding box acts as a surrogate for the relative location of the wells. 
            
                The dictionary is a nested dictionary, structured as follows:
                {'field_id_0': {"total_rois": [total_rois_0, total_rois_1, ...], 
                                "png_path": [png_path_0, png_path_1, ...], 
                                "first_well_coords": [sum_boxes_0, sum_boxes_1, ...]}}
                
                The nested structure exists because the entire nanowell is separated into 
                fields of view at each time point. The goal of well matching is to ensure that 
                the wells are segmented across the same location in each filed at each timepoint,
                therefore, the sub-dictionary for each field contains information about all the 
                images collected at different timepoints in a given field. 

                'total_rois' is a list of the number of ROIs segmented in each image at each time point.
                'png_path' is a list of the file paths to the images in the time series.
                'first_well_coords' is a list of the sum of the bounding boxes for the first well in each time point.

            well_dict: (dict) A dictionary containing the ROIs, bounding box coordinates, and time point identifier 
                for each well across all time points. This dictionary is updated if the user decides to overwrite
                the ROIs from a given field across timepoints with discordant segmentations. 
        Outputs:
            rewrite: (bool) A boolean indicating whether the user has decided to overwrite the ROIs from a given field.
            whole_image_dict: (dict) The updated dictionary containing the field level information for each time point.
        '''
        for field, vals in whole_image_dict.items():
            print(f'Checking field {field} for consistency across time points')
            total_rois = vals["total_rois"]
            png_paths = vals['png_path']
            #first_well_box = vals['first_well_coords']
            # Check if the number of ROIs is consistent across time points
            if not all(x == total_rois[0] for x in total_rois):
            # If ROIs are not consistent, find the time points that are discordant
                for i in range(len(total_rois)):
                    next((i for i, x in enumerate(total_rois[1:], 1) if x != total_rois[i-1]), -1)
                    print(f"Time point {i} has {total_rois[i]} ROIs")
                    # Identify the paths to the images that are discordant
                    time_point_1 = 'p'+str(i)
                    time_point_2 = 'p'+str(i+1)
                    plot_path_t1 = png_paths[i].replace('.png', '_validation.png')
                    plot_path_t2 = png_paths[i+1].replace('.png', '_validation.png')

            # sum_boxes = [np.sum(box) for box in first_well_box]
            # for i, (current, next) in enumerate(zip(sum_boxes, sum_boxes[1:])):
            #     if i < len(total_rois) - 1:
            #         current_rois = total_rois[i]
            #         print("Current Time Point: ", current_rois)
            #         next_rois = total_rois[i+1]
            #         print("Next Time Point: ", next_rois)
            #         if i < 10:
            #             time_point_1 = 'p0'+str(i)
            #         if i+1 < 10:
            #             time_point_2 = 'p0'+str(i+1)
            #         else:
            #             time_point_1 = 'p'+str(i)
            #             time_point_2 = 'p'+str(i+1)
            #         if abs(current - next) > 10 or current_rois != next_rois:
            #             print(f"Box {i} is different than box {i+1} by more than 10 pixels")
            #             # access validation plot images (from adjacent time points)
            #             roi_path = self.generate_rois_params.get('roi_path')
            #             validation_directory = roi_path + '/validation_plots'
            #             # Need new field str source from 'field' key in whole_image_dict
            #             field_str = field[9:]
            #             print("Time Points:" , time_point_1, time_point_2)
            #             plot_path_t1 = find_files(validation_directory, field_str, 'validation', time_point_1)
            #             plot_path_t2 = find_files(validation_directory, field_str, 'validation', time_point_2)
                        # Create comparison plot of the two images with labeled ROIs
            print("Path 1: ", plot_path_t1)
            print("Path 2: ", plot_path_t2)
            roi_path = self.generate_rois_params.get('roi_path')
            comparison_plot_path = roi_path + '/comparison_plots'
            if len(plot_path_t1) > 0 and len(plot_path_t2) > 0:
                generate_comparison_plot(plot_path_t1[0], 
                                        plot_path_t2[0], 
                                        field_of_view=field_str, 
                                        time_point_1=time_point_1,
                                        time_point_2=time_point_2, 
                                        save_path=comparison_plot_path)
            # User makes decision on whether to overwrite the ROIs of the image that had an incorrect segmentation
            user_input = input("Please view discordant ROIs in the comparison plot.\n"
                            "If working outside a Jupyter{ notebook, the comparison plot will be saved in the ROI directory.\n"
                            f"If ROI {i} is incorrect, enter (1) to overwrite the ROIs with the ROIs from {i+1}.\n"
                            f"If ROI {i+1} is incorrect, enter (2) to overwrite the ROIs with the ROIs from {i}.\n"
                            "If no action should be taken, enter (0).\n")
            # Response to user
            #TODO: Modify this to update the whole_image_dict, not the well_dict
            # 
            if user_input == '1':
                whole_image_dict[field][0][i] = whole_image_dict[field][0][i+1]
                whole_image_dict[field][1][i] = whole_image_dict[field][1][i+1]
                whole_image_dict[field][2][i] = whole_image_dict[field][2][i+1]
                rewrite = True
            elif user_input == '2':
                whole_image_dict[field][0][i+1] = whole_image_dict[field][0][i]
                whole_image_dict[field][1][i+1] = whole_image_dict[field][1][i]
                whole_image_dict[field][2][i+1] = whole_image_dict[field][2][i]
                rewrite = True
            else:
                continue
        return rewrite, whole_image_dict

    def match_wells(self,
                    whole_image_dict: dict,
                    well_dict: dict) -> None:
        pass

        return

    def segment_nanowells(self, 
                          model_path: str = None, 
                          model_type: str = 'vit_h', 
                          output_directory: str = None, 
                          convert_png: bool = False,
                          **kwargs) -> dict:
        '''
        Segment and crop the images in the experiment.
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

        Inputs:
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
            
        Outputs:
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
                {'field_id_0': {"total_rois": total_rois, 
                                "png_path": png_path, 
                                "first_well_coords": [sum_boxes]}
                'field_id_1': {"total_rois": total_rois,
                                "png_path": png_path,
                                "first_well_coords": [sum_boxes]}
        '''
        assert isinstance(model_path, str), "Model checkpoint path on local machine must be specified for segmentation. \
            Model checkpoints must be downloaded from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints."
        self.generate_rois_params = kwargs.get('generate_rois_params', {})
        self.plot_segmented_image_params = kwargs.get('plot_segmented_image_params', {})
        self.plot_masks_params = kwargs.get('plot_masks_params', {})
        # Extract well_location_tolerance to specify the allowed difference in the centroid of the bounding box
        # between well locations in adjacent time points. Default is 5 pixels.
        well_location_tolerance = kwargs.get('well_location_tolerance', 5)
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
            for i, j in enumerate(os.listdir(png_image_directory_path)):
                png_path=png_image_directory_path + '/' + j
                tif_path=tif_image_directory_path + '/' + j.rstrip('.png') + '.TIF'
                if begin_segmenting is True: # Initialize SamSegmenter instance
                    segmentation = ac.SamSegmenter(model_path=model_path, 
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
                                                                      img_idx=i,
                                                                      well_location_tolerance=well_location_tolerance)
        # Perform well-matching. If well locations are not consistent across time points,
        # the user will be prompted to verify the discordance.
        # TODO: Modify to be consistent with the new well_matching() logic
        if self.generate_rois_params.get('well_match') is True:
            print("Matching wells across time points")
            rewrite, whole_image_dict = self.match_wells(whole_image_dict, well_dict)
            if rewrite:
                well_dict = self.generate_image_dicts(len(roi), 
                                                    field_str, 
                                                    png_path, 
                                                    roi, 
                                                    box, 
                                                    centroids,
                                                    whole_image_dict, 
                                                    well_dict,
                                                    img_idx=i,
                                                    well_location_tolerance=well_location_tolerance)
        self.well_dict = well_dict
        self.whole_image_dict = whole_image_dict
        return whole_image_dict, well_dict



#TODO -- START OVER...
    def crop_nanowells(self,
                       output_directory: str = None):
        '''
        Crops the images well-wise across time points for an experiment to create a new directory for
        each well across time points. The cropped images will be saved in the output directory.
        Note that this function will crop all channels, as defined in the NanoExperiment.

            output_directory/
                cropped_tif_images/
                    field_id_0_channel_id_d0/
                        field_id_0_channel_id_0_well_id_0/
                            image_time_point_id_0_well_id_0.tif 
                            image_time_point_id_1_well_id_0.tif
                        ...
        '''
        segment_channel = str(self._channel_id) + str(self._segment_channel)
        if output_directory is None:
            cropped_tif_image_path = self._experiment_path + '/cropped_tif_images'
        else:
            cropped_tif_image_path = output_directory + '/cropped_tif_images'
        if not os.path.exists(cropped_tif_image_path):
            os.makedirs(cropped_tif_image_path)
        # Channels are indexed from 0
        for i in range(self._num_channels):
            print('Channel: ', i)
            for field_id_well_id, well_info in self.well_dict.items():
                print("Field ID and Well ID: ", field_id_well_id)
                # Create directory for each field and channel
                field_channel_path = cropped_tif_image_path + '/' + field_id_well_id[:3] + self._channel_id + str(i)
                # Get original TIF image path
                if not os.path.exists(field_channel_path):
                        os.makedirs(field_channel_path)
                # Select the well-level information
                for j, roi in enumerate(well_info[0]):
                    print("Methods and Properties:")
                    print(dir(roi))
                    # Create image path
                    # The channel will need to be changed, because the pngs were generated from the brightfield channel
                    png_path = well_info[2][j]
                    # Change the channel in the image path
                    png_path = png_path.replace(segment_channel, str(self._channel_id) + str(i))
                    print('PNG PATH: ', png_path)
                    field_channel_well_path = field_channel_path + '/' + field_id_well_id[4:]
                    if not os.path.exists(field_channel_well_path):
                        os.makedirs(field_channel_well_path)
                    og_tif_path = self._experiment_path + '/' + field_id_well_id[:3] + self._channel_id + str(i) + '/' + png_path.rstrip('.png') + '.TIF'
                    print('Original TIF path: ', og_tif_path)
                    # TODO: Figure out a different cropping mechanism, this IS NOT WORKING
                    cropped_tif_path = field_channel_well_path + '/' + png_path.rstrip('.png') + '.TIF'
                    # initialize imagej
                    ij = imagej.init('sc.fiji:fiji')
                    # load image
                    image = ij.io().open(og_tif_path)
                    if not isinstance(image, ImgPlus):
                        image = ImgPlus(image)
                    # get roi as rectangle
                    x = int(roi.left)
                    y = int(roi.top)
                    width = int(roi.widthd)
                    height = int(roi.heightd)
                    # Calculate max coordinates
                    x_max = x + width - 1
                    y_max = y + height - 1
                    # Create Java long arrays from Python lists
                    start = JArray(JLong)([x, y])
                    end = JArray(JLong)([x_max, y_max])
                    # Create interval
                    interval = FinalInterval(start, end)
                    # crop image
                    cropped_im = ij.op().run("net.imagej.ops.transform.crop.CropImgPlus", image, interval, False)
                    # Save TIF
                    ij.io().save(cropped_im, cropped_tif_path)
                    # Save png
                    cropped_png_path = field_channel_well_path + '/' + png_path
                    ij.io.save(cropped_im, cropped_png_path)
                    # Clean up
                    ij.dispose()
        return None
    
    