import math
import os
import time

import iohub.ngff as ngff
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import micro_dl.input.inference_dataset as inference_dataset
import micro_dl.torch_unet.utils.model as model_utils
import micro_dl.utils.aux_utils as aux_utils


def _pad_input(x: Tensor, num_blocks: int):
    """
    Zero-pads row and col dimensions of inputs to a multiple of 2**num_blocks

    :param torch.tensor x: input tensor

    :return torch.tensor x_padded: zero-padded x
    :return tuple pad_shape: shape x was padded by (left, top, right, bottom)
    """
    down_factor = 2**num_blocks
    sizes = [down_factor * math.ceil(s / down_factor) - s for s in x.shape[-2:]]
    pads = [(p // 2, p - p // 2) for p in sizes]
    pads = (pads[1][0], pads[0][0], pads[1][1], pads[0][1])
    x = TF.pad(x, pads)
    return x, pads




class TorchPredictor:
    """
    TorchPredictor class 
    TorchPredictor object handles all procedures involved with model inference.
    Utilizes an InferenceDataset object for reading data in from the given zarr store

    Params:
    :param dict torch_config: master config file
    """

    def __init__(self, torch_config, device=None) -> None:
        self.torch_config = torch_config

        self.zarr_dir = self.torch_config["zarr_dir"]
        self.network_config = self.torch_config["model"]
        self.dataset_config = self.torch_config["dataset"]
        self.inference_config = self.torch_config["inference"]

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
        model = model_utils.model_init(
            self.network_config,
            device=self.device,
            debug_mode=False,
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
        #FIXME update to new inference 
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
        Runs prediction on entire image field of view.
        If the input XY size is not compatible with the model
        (a multiple of :math:`2^{blocks}`),
        it will be padded with zeros on all sides for inference
        and cropped to the original size before output.
        Input must be either 4 or 5 dimensions, and output is returned with the
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
            img_tensor = aux_utils.ToTensor(device=self.device)(input_image)

        elif self.network_config["architecture"] == "2D":
            # Torch Unet 2D takes 2 spatial dims, handle lingering 1 in z dim
            if len(input_image.shape) != 4:
                raise ValueError(
                    f"2D unet must take 4D input data. Received {len(input_image.shape)}."
                    " Check preprocessing config."
                )
            img_tensor = aux_utils.ToTensor(device=self.device)(input_image)

        img_tensor, pads = _pad_input(img_tensor, num_blocks=model.num_blocks)
        pred = model(img_tensor, validate_input = False)
        return TF.crop(pred.detach().cpu(), *(pads[1], pads[0]) + input_image.shape[-2:]).numpy()

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
                            in the format written above
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
        self.dataset.channels = self.inference_config["input_channels"]
        print("Running inference: \n")
        i = 0
        for row_name, col_name, fov_name in tqdm(position_paths, position=0):
            shape, dtype = self.dataset.set_source_array(row_name, col_name, fov_name)
            output_position = self.output_writer.create_position(row_name, col_name, fov_name)
            output_array = output_position.create_zeros(
                name=["0"],
                shape = shape,
                dtype = dtype,
                chunks=(1,) * len(shape) - 2 + shape[-2:]
            )

            batch_size = shape[0] - (self.dataset_config["window_size"] - 1)
            dataloader = iter(DataLoader(self.dataset, batch_size=batch_size))
            
            description = "predicting " + str((row_name, col_name, fov_name, time_idx))
            for time_idx in tqdm(self.inference_config["time_indices"], desc=description, position=1, leave=False):
                    batch = next(dataloader)
                    batch_prediction = torch.squeeze(torch.swapaxes(self.predict_image(batch), 0, -3))
                    output_array[time_idx] = batch_prediction
            i += 1
        
        self.output_writer.close()
        print(f"Done! Time taken: {time.time() - start:.2f}s")
        print(f"Predictions and visualizations saved to {self.save_folder}")
