import os
import numpy as np
import fnmatch
import attiicc as ac
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

    def match_wells(self,
                    well_dict: dict) -> None:
        '''
        Match wells across time points. If the well locations are not consistent across time points,\
        the user will be prompted to verify the discordance and given the opportunity to overwrite\
        the ROIs of the image that had an incorrect segmentation with the ROIs from the adjacent time point.\n
        Inputs:
            well_dict: (dict) A dictionary containing the ROIs, bounding box coordinates,
                and time point identifier for each well across all time points.
                The dictionary is structured as follows:
                {'field_id_0_well_id_0': [[roi_0, roi_1, ...], 
                                        [box_0, box_1, ...], 
                                        [time_point_0, time_point_1, ...]], ...}
        Outputs:
            None
        '''
        field_set = set()
        for well, vals in well_dict.items():
            if well[:3] in field_set:
                continue
            else:
                field_set.add(well[:3])
                print(f'Checking well {well} for consistency across time points')
                boxes = vals[1]
                total_rois = vals[3]
                print(total_rois)
                sum_boxes = [np.sum(box) for box in boxes]
                for i, (current, next) in enumerate(zip(sum_boxes, sum_boxes[1:])):
                    print("hello")
                    print(i)
                    print(len(total_rois) - 1)
                    if i < len(total_rois) - 1:
                        current_rois = total_rois[i]
                        print("Current ROI: ", current_rois)
                        next_rois = total_rois[i+1]
                        print("Next ROI: ", next_rois)
                        if i < 10:
                            time_point_1 = 'p0'+str(i)
                        if i+1 < 10:
                            time_point_2 = 'p0'+str(i+1)
                        else:
                            time_point_1 = 'p'+str(i)
                            time_point_2 = 'p'+str(i+1)
                        if abs(current - next) > 10 or current_rois != next_rois:
                            print(f"Box {i} is different than box {i+1} by more than 10 pixels")
                            # access validation plot images (from adjacent time points)
                            roi_path = self.generate_rois_params.get('roi_path')
                            validation_directory = roi_path + '/validation_plots'
                            field_str = well[:3]
                            print("Time Points:" , time_point_1, time_point_2)
                            plot_path_t1 = find_files(validation_directory, field_str, 'validation', time_point_1)
                            plot_path_t2 = find_files(validation_directory, field_str, 'validation', time_point_2)
                            # Create comparison plot of the two images with labeled ROIs
                            print("Path 1: ", plot_path_t1)
                            print("Path 2: ", plot_path_t2)
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
                            if user_input == '1':
                                well_dict[well][0][i] = well_dict[well][0][i+1]
                                well_dict[well][1][i] = well_dict[well][1][i+1]
                                well_dict[well][2][i] = well_dict[well][2][i+1]
                            elif user_input == '2':
                                well_dict[well][0][i+1] = well_dict[well][0][i]
                                well_dict[well][1][i+1] = well_dict[well][1][i]
                                well_dict[well][2][i+1] = well_dict[well][2][i]
                            else:
                                continue
        return None

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
                                        [time_point_0, time_point_1, ...]],
                'field_id_0_well_id_1': [[roi_0, roi_1, ...],
                                        [box_0, box_1, ...],
                                        [time_point_0, time_point_1, ...]]}
            
        '''
        assert isinstance(model_path, str), "Model checkpoint path on local machine must be specified for segmentation. \
            Model checkpoints must be downloaded from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints."
        self.generate_rois_params = kwargs.get('generate_rois_params', {})
        self.plot_segmented_image_params = kwargs.get('plot_segmented_image_params', {})
        self.plot_masks_params = kwargs.get('plot_masks_params', {})
        if output_directory is None:
            roi_path = self._experiment_path + '/ROI'
        else: 
            roi_path = output_directory + '/ROI'
        well_dict = {}
        if not os.path.exists(roi_path):
            os.makedirs(roi_path)
        # Get path to image for segmentation
        channel_str = self._channel_id + str(self._segment_channel)
        # Locate and/or create PNG images needed for segmentation from brightfield images
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
            whole_image_dict = {}
            whole_image_dict[field_str] = {"roi_lengths": [], "png_path": []}
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
                roi, box = segmentation.generate_rois(**self.generate_rois_params)
                whole_image_dict[field_str]["roi_lengths"].append(len(roi))
                whole_image_dict[field_str]["png_path"].append(png_path)
                # Save ROIs and boxes in dictionary for each well
                for result_index in range(len(roi)):
                    total_rois = len(roi)
                    well_name = f'{field_str}_well{result_index}'
                    png_name = png_path.split('/')[-1]
                    # time_point_index = png_path.find(self._time_point_id)
                    # if time_point_index != -1:
                    #     if self._time_point_leading_zero:
                    #         time_point = png_path[time_point_index:time_point_index + 3]
                    #     else:
                    #         time_point = png_path[time_point_index:time_point_index + 2]
                    if well_name not in well_dict:
                        well_dict[well_name] = [[roi[result_index]], [box[result_index]], [png_name], [total_rois]]
                    else:
                        well_dict[well_name] = [well_dict[well_name][0] + [roi[result_index]], 
                                                well_dict[well_name][1] + [box[result_index]],
                                                well_dict[well_name][2] + [png_name], 
                                                well_dict[well_name][3] + [total_rois]]
                    if 'well0' in well_name:
                        sum_boxes = [np.sum(box) for box in well_dict[well_name][1]]
                        whole_image_dict[well_name] = sum_boxes
        # Perform well-matching. If well locations are not consistent across time points,
        # the user will be prompted to verify the discordance.
        if self.generate_rois_params.get('well_matching'):
            self.match_wells(well_dict)
        return whole_image_dict


    def crop_nanowells(self, output_directory: str = None):
        '''
            output_directory/
                cropped_tif_images/
                    field_id_0_channel_id_0_well_id_0/
                        image_time_point_id_0_well_id_0.tif 
                        image_time_point_id_1_well_id_0.tif
                        ...
        '''
        if output_directory is None:
            cropped_tif_image_path = self._experiment_path + '/cropped_tif_images'
        else:
            cropped_tif_image_path = output_directory + '/cropped_tif_images'
        if not os.path.exists(cropped_tif_image_path):
            os.makedirs(cropped_tif_image_path)

        pass
        return None