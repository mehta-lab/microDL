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
        self.y_window_size, self.x_window_size = inference_config["window_size"][-2:]
        
        self.data_plate = ngff.open_ome_zarr(
            store_path=zarr_dir,
            layout='hcs',
            mode='r',
        )
        
    def __len__(self):
        """Returns the number of positions specified for inference"""
        return len(self.position_mapping)
    
    def __getitem__(
        self, 
        row_idx, 
        col_idx, 
        fov_idx, 
        time_idx, 
        channel_ids, 
        slice_range,
        return_norm_statistics=False,
        ):
        """
        Returns the requested channels and slices of the data at position 
        specified by row, col, and fov indices and timestep specified by 
        time_idx.
        
        Note: if multiple channels requested will return tensor of shape ()

        :param int row_idx: row_index of position
        :param int col_idx: colum index of position
        :param int fov_idx: field of view index
        :param int time_id: timestep index to retrieve
        :param int, tuple(int) channel_ids: channel indices to retrieve
        :param int, tuple(int) slice_range: range slices  to retrieve in 
                        format (int, int)
        :param bool return_norm_statistics: if True, returns normalization
                        statistics for each requested channel in order
                        
        :return torch.Tensor convert: requested image stack as a tensor
        :return list norm_statistics: (optional) list of normalization statistics
                        dicts for each channel in the returned array
        """
        position = self.data_plate["/".join(map(str, [row_idx, col_idx, fov_idx]))]
        
        shape = position.data.shape
        assert slice_range[0] > 0 and slice_range[1] <= position.data.slices, (
            f"Requested center slice with slice range {slice_range} is out of bounds"
            f" for data with shape {shape}."
        )
        if isinstance(channel_ids, int):
            channel_ids = [channel_ids]
        #TODO use channel names instead of IDS (needs to be done elsewhere in code)
        # channel_names = self.data_plate.channel_names
        
        # looping instead of numpy slicing to allow for discontinuous channel selections
        data = []
        norm_statistics = []
        self.item_chan_names = []
        for channel in channel_ids:
            channel_data = position.data[
                time_idx,
                channel, 
                slice_range[0]: slice_range[1], 
                :self.y_window_size, 
                :self.x_window_size
                ]
            
            channel_name = position.channel_names[channel]
            channel_norm_statistics = self._get_normalization_statistics(
                row_idx, col_idx, fov_idx, channel_name
            )
            norm_statistics.append(channel_norm_statistics)
            self.item_chan_names.append(channel_name)
            
            if self.inference_config.get("normalize_inputs"):
                channel_data = self._normalize(channel_data, channel_norm_statistics)
            data.append(channel_data)
            
        data = np.stack(data, axis=0)
        
        convert = aux_utils.ToTensor(self.inference_config["device"])(data)
        if return_norm_statistics:
            return convert, channel_norm_statistics
        else:
            return convert
    
    def _get_normalization_statistics(self, row_idx, col_idx, fov_idx, channel_name):
        """
        Gets and returns the normalization statistics stored in the .zattrs of a 
        specific position. 
        
        :param int row_idx: row_index of position
        :param int col_idx: colum index of position
        :param int fov_idx: field of view index
        """
        position = self.data_plate[row_idx][col_idx][fov_idx]
        normalization_metadata = position.zattrs["normalization"]
        return normalization_metadata[channel_name]
    
    def _normalize(self, data, normalization_meta, denorm=False):
        """
        Given the normalization meta for a specific chunk of data in the format:

        "dataset_statistics": {
            "iqr": some iqr,
            "mean": some mean,
            "median": some median,
            "std": some std
        },
        "fov_statistics": { (optional)
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
        #TODO Update FOV statistics location to work with new scheme:
        # Dataset statistics located in plate attrs file
        
        
        norm_type = self.dataset_config["normalization"]["type"]
        norm_scheme = self.dataset_config["normalization"]["scheme"]
        norm_function = normalize.unzscore if denorm else normalize.zscore
        if norm_scheme == "FOV":
            statistics = normalization_meta["fov_statistics"]
        elif norm_scheme == "dataset":
            statistics = normalization_meta["dataset_statistics"]
        else:
            return data

        if norm_type == "median_and_iqr":
            normalized_data = norm_function(
                data,
                zscore_median=statistics["median"],
                zscore_iqr=statistics["iqr"],
            )
        elif norm_type == "mean_and_std":
            normalized_data = norm_function(
                data,
                zscore_median=statistics["mean"],
                zscore_iqr=statistics["std"],
            )

        return normalized_data
    
    def _normalize_multichan(self, data, normalization_meta, denorm=False):
        """
        Given the list normalization meta for a specific multi-channel chunk of 
        data whose elements are each dicts of the form:
        
        "dataset_statistics": {
            "iqr": some iqr,
            "mean": some mean,
            "median": some median,
            "std": some std
        },
        "fov_statistics": { (optional)
            "iqr": some iqr,
            "mean": some mean,
            "median": some median,
            "std": some std
        }
        
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