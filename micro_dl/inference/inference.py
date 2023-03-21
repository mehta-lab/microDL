import datetime
import time
import numpy as np
import os
import zarr
import pathlib

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import micro_dl.torch_unet.utils.model as model_utils
import micro_dl.input.dataset as ds
import micro_dl.utils.cli_utils as cli_utils
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.io_utils as io_utils
import micro_dl.inference.evaluation_metrics as inference_metrics
import micro_dl.input.inference_dataset as inference_dataset
import iohub.ngff as ngff


class TorchPredictor:
    """
    TorchPredictor object handles all procedures involved with model inference.
    Utilizes an InferenceDataset object for reading data in from the given zarr store

    Params:
    :param dict torch_config: master config file
    """

    def __init__(self, torch_config, device=None) -> None:
        self.torch_config = torch_config

        self.zarr_dir = self.torch_config["zarr_dir"]
        self.network_config = self.torch_config["model"]
        self.training_config = self.torch_config["training"]
        self.dataset_config = self.torch_config["dataset"]
        self.inference_config = self.torch_config["inference"]
        self.preprocessing_config = self.torch_config["preprocessing"]

        self.inference_metrics = {}
        self.model = None
        if device:
            self.device = device
        else:
            self.device = self.inference_config["device"]

        # init dataset
        self.dataset = inference_dataset.TorchInferenceDataset(
            zarr_dir=self.zarr_dir,
            dataset_config=self.dataset_config,
            inference_config=self.inference_config,
        )
        
        # get directory for inference figure saving
        self.get_save_location()
        self.read_model_meta()

    def load_model(self, init_dir=True) -> None:
        """
        Initializes a model according to the network configuration dictionary used
        to train it, and loads the parameters saved in model_dir into the model's state dict.

        :param str init_dir: directory containing model weights and biases (should be true)
        """
        debug_mode = False
        if "debug_mode" in self.network_config:
            debug_mode = self.network_config["debug_mode"]

        model = model_utils.model_init(
            self.network_config,
            device=self.device,
            debug_mode=debug_mode,
        )

        if init_dir:
            model_dir = self.inference_config["model_dir"]
            readout = model.load_state_dict(
                torch.load(model_dir, map_location=self.device)
            )
            print(f"PyTorch model load status: {readout}")
        self.model = model

    def read_model_meta(self, model_dir=None):
        """
        Reads the model metadata from the given model dir and stores it as an attribute.
        Use here is to allow for inference to 'intelligently' infer it's configuration
        parameters from a model directory to reduce hassle on user.

        :param str model_dir: global path to model directory in which 'model_metadata.yml'
                        is stored. If not specified, infers from inference config.
        """
        if not model_dir:
            model_dir = os.path.dirname(self.inference_config["model_dir"])

        model_meta_filename = os.path.join(model_dir, "model_metadata.yml")
        self.model_meta = aux_utils.read_config(model_meta_filename)

    def get_save_location(self):
        """
        Sets save location as specified in config files.

        Note: for save location to be different than training save location,
        not only does inference/save_preds_to_model_dir need to be False,
        but you must specify a new location in inference/custom_save_preds_dir

        This is to encourage saving model inference with training models.

        """
        # TODO Change the functionality of saving to put inference in the actual
        # train directory the model comes from. Not a big fan

        model_dir = os.path.dirname(self.inference_config["model_dir"])
        save_to_train_save_dir = self.inference_config["save_preds_to_model_dir"]

        if save_to_train_save_dir:
            save_dir = model_dir
        elif "custom_save_preds_dir" in self.inference_config:
            custom_save_dir = self.inference_config["custom_save_preds_dir"]
            save_dir = custom_save_dir
        else:
            raise ValueError(
                "Must provide custom_save_preds_dir if save_preds_to"
                "_model_dir is False."
            )

        now = aux_utils.get_timestamp()
        self.save_folder = os.path.join(save_dir, f"inference_results_{now}")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def predict_image(self, input_image, model=None,):
        """
        Runs prediction on entire image field of view. xy size is configurable, but it must be
        a power of 2. Input must be either 4 or 5 dimensions, and output is returned with the
        same dimensionality as given in input.

        Params:
        :param numpy.ndarray/torch.Tensor input_image: input image or image stack on which
                                                        to run prediction
        :param Torch.nn.Module model: trained model to use for prediction
        
        :return np.ndarray prediction: prediction
        """
        assert (
            self.model != None or model != None
        ), "must specify model in init or prediction call"
        assert 5 - len(input_image.shape) <= 1, (
            f"input image has {len(input_image.shape)} dimensions"
            ", either 4 (2D) or 5 (2.5D) required."
        )
        if model == None:
            model = self.model

        if self.network_config["architecture"] == "2.5D":
            if len(input_image.shape) != 5:
                raise ValueError(
                    f"2.5D unet must take 5D input data. Received {len(input_image.shape)}."
                    " Check preprocessing config."
                )
            img_tensor = ds.ToTensor(device=self.device)(input_image)

        elif self.network_config["architecture"] == "2D":
            # Torch Unet 2D takes 2 spatial dims, handle lingering 1 in z dim
            if len(input_image.shape) != 4:
                raise ValueError(
                    f"2D unet must take 4D input data. Received {len(input_image.shape)}."
                    " Check preprocessing config."
                )
            img_tensor = ds.ToTensor(device=self.device)(input_image)

        pred = model(img_tensor, validate_input = False, pad_nonmultiple_input = True)
        return pred.detach().cpu().numpy()

    def _get_positions(self):
        """
        Logic function for determining the paths to positions requested for inference
        
        Positions should be specified in config in format:
            positions:
                row #:
                    col #: [pos #, pos #, ...]
                    col #: [pos #, pos #, ...]
                        .
                        .
                        .
                    .
                    .
                    .
        where row # and col # together indicate the well on the plate, and pos # indicates
        the number of the position in the well.
        
        If position paths are not specified in the config, requires that some portion of
        the data split generated during training is specified. If data portion is specified, 
        extracts the positions from the directory of the inference model. 

        :return dict positions: Returns dictionary tree specifying all the positions 
                            in the format:
                            positions = {
                                row #:
                                    {
                                        col #: [pos #, pos #, ...],
                                        col #: [pos #, pos #, ...],
                                        ...
                                    },
                                ...
                            }
        """
        # Positions are specified
        if isinstance(self.inference_config["positions"], dict):
            print("Predicting on positions specified in inference config.")
            return self.inference_config["positions"]
        elif not isinstance(self.inference_config["positions"], str):
            raise AttributeError(
                "Invalid 'positions'. Expected one of {str, dict}"
                f" but got {self.inference_config['positions']}"
            )

        #Positions are unspecified and need to be read
        model_dir = os.path.dirname(self.inference_config["model_dir"])
        data_split_file = os.path.join(model_dir, "data_splits.yml")
        split_section = self.inference_config["positions"]

        if os.path.exists(data_split_file):
            print(f"Predicting on {split_section} data split found in "
                  "model directory.")
            data_splits = aux_utils.read_config(data_split_file)
            timestamps = list(data_splits.keys())
            timestamps.sort(reverse=True)
            most_recent_split = timestamps[0]

            positions = data_splits[most_recent_split][split_section]
            return positions
        else:
            raise ValueError(
                f"Specified prediction on data split "
                f"'{self.inference_config['positions']}'"
                f" but no data_splits.yml file found in dir {model_dir}.\n"
            )

    def _validate_ids(self, ids, shape, name):
        """ 
        Ensures usage of all ids if -1 is passed, and if not, that all ids are
        represented in data
        
        :param int, tuple(int) ids: tuple of ids or -1
        :param int shape: shape of data in that dimension
        :param str name: name of dimension
        """
        if isinstance(ids, int) and ids == -1:
            if name == "slice":
                return (0, shape) 
            else:
                return tuple(range(shape))
            
        assert max(ids) > shape, (
            f"{name} indices {[i for i in ids if i >= shape]}"
            f"not in array with {shape} {name} indices"
        )
        return ids
    
    
    def run_inference(self):
        """
        Performs inference on the entire validation dataset.

        Model inputs are normalized before predictions are generated. Normalized and denormalized
        copies are saved, the latter being for visual evaluation, the former for metrics in 
        evaluation.
        """

        # init io and saving
        start = time.time()
        self.log_writer = SummaryWriter(log_dir=self.save_folder)
        self.output_writer = ngff.open_ome_zarr(
                os.path.join(self.save_folder, "preds.zarr"),
                layout="hcs",
                mode="w-",
                channel_names=self.inference_config["input_channels"],
            )
        self.model.eval()

        # generate list of position tuples from dictionary for iteration
        positions_dict = self._get_positions()
        position_paths = []
        for row_k, row_v in positions_dict.items():
            for well_k, well_v in row_v.items():
                fov_path_tuples = [(row_k, well_k, pos_k) for pos_k in well_v]
                position_paths.extend(fov_path_tuples)
        
        # run inference on each position
        print("Running inference: \n")
        i = 0
        depth = self.dataset_config["window_size"][0]
        for row_name, col_name, fov_name in position_paths:
            timepoint_preds = []
            #TODO This currently holds the *entire* position in memory, 
            # which is a problem for very large positions.
            # Should be able to be fixed with api exposed by
            # merging of https://github.com/czbiohub/iohub/pull/87
            for time_idx in self.inference_config["time_indices"]:
                process_string = "predicting " + str((row_name, col_name, fov_name, time_idx))
                cli_utils.show_progress_bar(
                    position_paths, i, process=process_string
                )
                
                pred_slices = []
                for center_slice in self.inference_config["center_slice_indices"]:
                    # load and predict data by slice
                    start, end = center_slice - depth//2, center_slice + depth//2 + 1
                    input_, norm_statistics = self.dataset.__getitem__(
                        row_idx=row_name,
                        col_idx=col_name,
                        fov_idx=fov_name,
                        time_idx=time_idx,
                        channel_ids=self.inference_config["input_channels"],
                        slice_range=(start, end),
                        return_norm_statistics=True
                    )
                    # FIXME: use dataloader to do batch predictions
                    prediction = self.predict_image(input_.unsqueeze_(0))
                    
                    # visualization logging
                    if self.inference_config["log_tensorboard"]:
                        denormed_prediction = self.dataset._normalize_multichan(
                            prediction, norm_statistics
                        )
                        channels = " + ".join(self.dataset.item_chan_names)
                        for i in range(denormed_prediction.shape[-4]):
                            self.log_writer.add_images(
                                tag=f"{row_name}.{col_name}.{fov_name}.{time_idx}"\
                                    f"/prediction channel {i} | input {channels}"\
                                    " | z{start}-{end}",
                                img_tensor=torch.tensor(denormed_prediction[0,i]),
                                dataformats="hw",
                            )
                    
                    pred_slices.append(prediction)
                timepoint_preds.append(np.stack(pred_slices, axis=0))
                
            #write position to an output zarr store
            position_preds = np.stack(timepoint_preds, axis=0)
            output_position = self.output_writer.create_position(row_name, col_name, fov_name)
            output_position["0"] = position_preds.squeeze(axis=(0, 1))
            
            i += 1
        
        self.output_writer.close()
        print(f"Done! Time taken: {time.time() - start:.2f}s")
        print(f"Predictions and visualizations saved to {self.save_folder}")
