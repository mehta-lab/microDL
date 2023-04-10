import iohub.ngff as ngff
import numpy as np
import os
from torch.utils.data import Dataset
import zarr

import micro_dl.utils.normalize as normalize
import micro_dl.utils.aux_utils as aux_utils

class TorchInferenceDataset(Dataset):
    """
    Based off of torch.utils.data.Dataset:
        - https://pytorch.org/docs/stable/data.html

    Custom dataset class for used for inference. Lightweight, dependent upon IOhub to
    read in data from an NGFF-HCS compatible zarr store and perform inference. 
    
    Predictions are written back to a zarr store inside the model directory, unless
    specified elsewhere.
    """
    def __init__(
        self,
        zarr_dir,
        dataset_config,
        inference_config,
    ):
        """
        Initiate object for selecting and passing data to model for inference.

        :param str zarr_dir: path to zarr store
        :param dict dataset_config: dict object of dataset_config   
        :param dict inference_config: dict object of inference_config       
        """
        self.zarr_dir = zarr_dir
        self.dataset_config = dataset_config
        self.inference_config = inference_config
        
        self.sample_depth, self.y_window_size, self.x_window_size = inference_config["window_size"]
        self.channels = None
        self.timesteps = None
        
        self.source_array = None
        self.data_plate = ngff.open_ome_zarr(
            store_path=zarr_dir,
            layout='hcs',
            mode='r',
        )
    
    def __len__(self):
        """Returns the number of valid center slices in position * number of timesteps"""
        return len(self.timesteps) * (self.source.data.shape[-3] - (self.sample_depth - 1))
    
    def __getitem__(self, idx):
        """
        Returns the requested channels and slices of the data at the current
        source array.
        
        Note: idx indexes into a mapping of timestep and center-z-slice. For example
        if timestep 2 and z slice 4 of 10 is requested, idx should be:
            2*10 + 4 = 24

        :param int idx: index in timestep & center-z-slice mapping
                        
        :return torch.Tensor data: requested image stack as a tensor
        :return list norm_statistics: (optional) list of normalization statistics
                        dicts for each channel in the returned array
        """
        # idx -> time & center idx mapping
        shape = self.source_position.data.shape
        timestep, center_idx = idx // shape[-3], (idx % shape[-3]) + self.sample_depth // 2
        
        # slice range depend on center slice and depth
        start_slice = center_idx - self.sample_depth // 2, 
        end_slice = center_idx + 1 + self.sample_depth // 2
        if isinstance(self.channels, int):
            self.channels = [self.channels]
        
        # retrieve data from selected channels
        channel_indices = [self.source_position.channel_names.index(c) for c in self.channels]
        data = self.source_position.data[timestep, channel_indices, start_slice : end_slice, ...]
        norm_statistics = [self._get_normalization_statistics(c) for c in self.channels]
        
        #normalize and convert
        if self.inference_config.get("normalize_inputs"):
                data = self._normalize_multichan(data, norm_statistics)
        data = aux_utils.ToTensor(self.inference_config["device"])(data)
        
        return data, norm_statistics
    
    def set_source_array(self, row, col, fov, timesteps = None, channels = None):
        """
        Sets the source array in the zarr store at zarr_dir that this 
        dataset should pull from when __getitem__ is called.
        
        :param str/int row_name: row_index of position
        :param str/int col_name: colum index of position
        :param str/int fov_name: field of view index
        :param tuple(int) time_id: (optional) timestep indices to retrieve
        :param tuple(str) channels: (optional) channel indices to retrieve
        
        :return tuple shape: shape of expected output from this source
        :return type dtype: dtype of expected output from this source
        """
        row, col, fov = map(str, [row, col, fov])
        self.source_position = self.data_plate[row][col][fov]
        
        self.timesteps = tuple(range(self.source.data.shape[0]))
        if timesteps:
            self.timesteps = timesteps
        
        channel_ids = tuple(range(self.source.data.shape[1]))
        self.channels = self.data_plate.channel_names(channel_ids)
        if channels:
            self.channels = channels
        
        return (len(self.timesteps), len(self.channels)) + self.source_position.data.shape[-3:], self.source_position.dtype
    
    def _get_normalization_statistics(self, channel_name):
        """
        Gets and returns the normalization statistics stored in the .zattrs of a 
        specific position. 
        
        :param str channel_name: name of channel
        """
        if self.dataset_config["normalization"]["scheme"] == "dataset":
            normalization_metadata = self.data_plate.zattrs["normalization"]
        else:
            normalization_metadata = self.source_position.zattrs["normalization"]
        return normalization_metadata[channel_name]
    
    def _normalize_multichan(self, data, normalization_meta, denorm=False):
        """
        Given the list normalization meta for a specific multi-channel chunk of 
        data whose elements are each dicts of normalization statistics.
        
        performs normalization on the entire stack as dictated by parameters in
        dataset_config.

        :param np.ndarray data: 4d numpy array (c, z, y, x)
        :param list normalization_meta: list of channel norm statistics for array 
        
        :param np.ndarray normalized_data: normalized 4d numpy array (c,z,y,x)
        """
        all_data = []
        for i, channel_norm_meta in enumerate(normalization_meta):
            channel_data = data[i]
            normed_channel_data = self._normalize(
                channel_data, 
                channel_norm_meta,
                denorm=denorm,
            )
            all_data.append(normed_channel_data)
        
        return np.stack(all_data, axis=0)
    
    def _normalize(self, data, normalization_meta, denorm=False):
        """
        Given the normalization meta for a specific chunk of data in the format:
        {
            "iqr": some iqr,
            "mean": some mean,
            "median": some median,
            "std": some std
        }

        zscores or un-zscores the data based upon the metadata and 'denorm'

        :param np.ndarray data: 3d un-normalized input data
        :param dict normalization_meta: dictionary of statistics containing precomputed 
                                    norm values for dataset and FOV
        :param bool denorm: Whether to apply or revert zscoring on this data

        :return np.ndarray normalized_data: denormed data of input data's shape and type
        """
        norm_type = self.dataset_config["normalization"]["type"]
        norm_function = normalize.unzscore if denorm else normalize.zscore

        if norm_type == "median_and_iqr":
            normalized_data = norm_function(
                data,
                zscore_median=normalization_meta["median"],
                zscore_iqr=normalization_meta["iqr"],
            )
        elif norm_type == "mean_and_std":
            normalized_data = norm_function(
                data,
                zscore_median=normalization_meta["mean"],
                zscore_iqr=normalization_meta["std"],
            )

        return normalized_data