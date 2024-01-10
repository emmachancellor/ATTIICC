import os
import numpy as np
import attiicc as ac
from experiment_utils import convert_tif_to_png

class NanoExperiment:
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
            field_leading_zero: bool = False
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
        Outputs:
            None
        '''
        assert isinstance(experiment_path, str), 'Experiment_path must be a string'
        assert isinstance(field_id, str), 'field_id must be a string specifying the field of view identifier, Ex. \'f\''
        assert isinstance(num_fields, int), 'num_fields must be an integer specifying the number of fields of view'
        assert isinstance(channel_id, str), 'channel_id must be a string specifying the channel identifier, Ex. \'c\''
        assert isinstance(num_channels, int), 'num_channels must be an integer specifying the number of channels'
        assert isinstance(time_point_id, str), 'time_point_id must be a string specifying the time point identifier, Ex. \'t\''
        assert isinstance(num_time_points, int), 'num_time_points must be an integer specifying the number of time points'
        self._experiment_path = experiment_path
        self._field_id = field_id
        self._num_fields = num_fields
        self._channel_id = channel_id
        self._num_channels = num_channels
        self._time_point_id = time_point_id
        self._segment_channel = segment_channel
        self._num_time_points = num_time_points
        self.structure = print(f'Experiment Structure /n segment_channel: {self._segment_channel} /n field_id: {self.field_id} /n field_num: {self.num_fields} /n \ 
                               field_leading_zero: {self._field_leading_zero} /n
                               channel_id: {self.channel_id} /n channel_num: {self.num_channels} /n \ 
                               time_point_id: {self.time_point_id} /n time_point_num: {self.num_time_points}')
        self._field_leading_zero = field_leading_zero

    @property
    def field_leading_zero(self) -> bool:
        return self._field_leading_zero
    
    @field_leading_zero.setter
    def field_leading_zero(self, field_leading_zero) -> None:
        self._field_leading_zero = field_leading_zero

    @property
    def structure(self) -> str:
        return self.structure
    
    @structure.setter
    def structure(self, field_id,
                  num_fields,
                  channel_id,
                  num_channels,
                  time_point_id,
                  num_time_points,
                  segment_channel) -> None:
        '''
        Set the structure of the experiment.
        '''
        self._field_id = field_id
        self._num_fields = num_fields
        self._channel_id = channel_id
        self._num_channels = num_channels
        self._time_point_id = time_point_id
        self._num_time_points = num_time_points
        self._segment_channel = segment_channel
        self.structure = print(f'Experiment Structure /n field_id: {self.field_id} /n field_num: {self.num_fields} /n \
                               field_leading_zero: {self._field_leading_zero} /n 
                               channel_id: {self.channel_id} /n channel_num: {self.num_channels} /n \ 
                               time_point_id: {self.time_point_id} /n time_point_num: {self.num_time_points}')

    def segment_and_crop(self, model_path: str = None, model_type: str = 'vit_h', output_directory: str = None, 
                            convert_png: bool = False) -> None:
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
                cropped_images/
                    field_id_0_channel_id_0_well_id_0/
                        image_time_point_id_0_well_id_0.tif 
                        image_time_point_id_1_well_id_0.tif
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
            model_path: (str) The path to the model checkpoint. This must be downloaded
                from Meta on a user's local machine. Checkpoints can be downloaded from \
                https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
            model_type (str, optional): Specify the sam model type to load.
                Default is "vit_h". Can use "vit_b", "vit_l", or "vit_h".
            output_directory: (str) The path to the output directory. This will house
                subdirectories for the ROIs and cropped images.
            convert_png: (bool) If True, the TIF images will be converted to PNG and saved in a new directory.
                These images will be used for segmentation. If PNG images exist,
                the directory must follow the same structure as the TIF directory, with 
                the directory extension 'png.' For example, '/experiment_path/field_id_0_channel_id_0_png/'.
                Default is False.
            
        '''
        assert isinstance(model_path, str), "Model checkpoint path on local machine must be specified for segmentation. \
            Model checkpoints must be downloaded from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints."
        roi_path = output_directory + '/ROI'
        cropped_image_path = output_directory + '/cropped_images'
        if not os.path.exists(roi_path):
            os.makedirs(roi_path)
        if not os.path.exists(cropped_image_path):
            os.makedirs(cropped_image_path)
        # Get path to image for segmentation
        channel_str = str(self._segment_channel)
        for f in range(self._num_fields):
            if self._field_leading_zero and f < 10:
                field_str = '0' + str(f)
            else:
                field_str = str(f)
            tif_image_directory_path = f'{self._experiment_path}/{field_str}{channel_str}'
            # Do this if we can't find PNGs, but look for them first and don't require a PNG bool
            if convert_png:
                png_image_directory_path = convert_tif_to_png(tif_image_directory_path)
            # Segment images
            for i, j in enumerate(os.listdir(png_image_directory_path)):
                image_path=png_image_directory_path + '/' + j
                tif_path=tif_image_directory_path + '/' + j.rstrip('.png') + '.TIF'
                if i == 0:
                    segmentation = ac.SamSegmenter(model_path=model_path, 
                                                   model_type=model_type, 
                                                   image_path=image_path,
                                                   tif_path=tif_path)
                else:
                    segmentation.image_path = (image_path, tif_path)
                #TODO:Save the segmentation results into dictionary ... 


            
         



