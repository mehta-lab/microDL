import datetime
import time
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import micro_dl.torch_unet.utils.model as model_utils
import micro_dl.torch_unet.utils.dataset as ds
import micro_dl.torch_unet.utils.io as io_utils


class TorchPredictor:
    """
    TorchPredictor object handles all procedures involved with model inference.
    The trainer uses a the model.py and dataset.py utility modules to instantiate and load a model
    and validation data for inference using a gunpowder backend.

    Params:
    :param torch_config
    """

    def __init__(self, torch_config) -> None:
        self.torch_config = torch_config

        self.zarr_dir = self.torch_config["zarr_dir"]
        self.network_config = self.torch_config["model"]
        self.training_config = self.torch_config["training"]
        self.dataset_config = self.torch_config["dataset"]
        self.inference_config = self.torch_config["inference"]

        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None
        self.current_dataloader = None

        self.model = None
        self.device = self.inference_config["device"]

        # get directory for inference figure saving
        self.get_save_location()

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
            device=self.training_config["device"],
            debug_mode=debug_mode,
        )

        if init_dir:
            model_dir = self.network_config["model_dir"]
            readout = model.load_state_dict(torch.load(model_dir))
            print(f"PyTorch model load status: {readout}")
        self.model = model

    def generate_dataloaders(self, train_key=None, test_key=None, val_key=None) -> None:
        """
        Helper that generates validation torch dataloaders for loading validation samples
        with a gunpowder backend. Note that a train/test/val data split must have already been
        recorded in the zarr_dir zarr store's top level .zattrs metadata during training in the
        following format in order for this method to work:
            "data_split_positions":{
                "train": -train positions (list[int])-,
                "test": -test positions (list[int])-,
                "val": -val positions (list[int])-,
            }

        Dataloaders are set to class variables. Dataloaders correspond to one multi-zarr
        dataset each. Each dataset's access key will determine the data array (by type)
        it calls at the well-level.

        If keys unspecified, defaults to the first available data array at each well

        :param int or gp.ArrayKey train_key: key or index of key to data array for training
                                            in training dataset
        :param int or gp.ArrayKey test_key: key or index of key to data array for testing
                                            in testing dataset
        :param int or gp.ArrayKey val_key: key or index of key to data array for validation
                                            in validation dataset
        """
        assert self.torch_config != None, (
            "torch_config must be specified in object" "initiation "
        )

        # init datasets
        workers = 0
        if "num_workers" in self.training_config:
            workers = self.training_config["num_workers"]

        torch_data_container = ds.InferenceDatasetContainer(
            zarr_dir=self.zarr_dir,
            train_config=self.training_config,
            network_config=self.network_config,
            dataset_config=self.dataset_config,
            device=self.device,
            workers=workers,
            use_recorded_split=True,
        )
        train_dataset_list = torch_data_container["train"]
        test_dataset_list = torch_data_container["test"]
        val_dataset_list = torch_data_container["val"]

        # initalize dataset keys
        train_key = 0 if train_key == None else train_key
        test_key = 0 if test_key == None else test_key
        val_key = 0 if val_key == None else val_key
        for train_dataset in train_dataset_list:
            train_dataset.use_key(train_key)
        for test_dataset in test_dataset_list:
            test_dataset.use_key(test_key)
        for val_dataset in val_dataset_list:
            val_dataset.use_key(val_key)

        # init dataloaders
        self.train_dataloader = DataLoader(
            dataset=ConcatDataset(train_dataset_list), shuffle=True
        )
        self.test_dataloader = DataLoader(
            dataset=ConcatDataset(test_dataset_list), shuffle=True
        )
        self.val_dataloader = DataLoader(
            dataset=ConcatDataset(val_dataset_list), shuffle=True
        )

    def select_dataloader(self, name="val"):
        """
        Selects dataloader from train, test, val

        :param name: _description_, defaults to "val"
        :type name: str, optional
        """
        assert self.train_dataloader and self.test_dataloader and self.val_dataloader, (
            "Dataloaders " " must be initated. Try 'object_name'.generate_dataloaders()"
        )
        assert name in {
            "train",
            "test",
            "val",
        }, "name must be one of 'train', 'test', 'val'"
        if name == "val":
            self.current_dataloader = self.val_dataloader
        if name == "test":
            self.current_dataloader = self.test_dataloader
        if name == "train":
            self.current_dataloader = self.train_dataloader

    def get_save_location(self):
        """
        Sets save location as specified in config files.

        Note: for save location to be different than training save location,
        not only does inference/save_preds_to_model_dir need to be False,
        but you must specify a new location in inference/save_preds_dir

        This is to encourage saving model inference with training models.

        """
        # TODO Change the functionality of saving to put inference in the actual
        # train directory the model comes from. Not a big fan
        train_save_dir = self.training_config["save_dir"]
        save_to_train_save_dir = self.inference_config["save_preds_to_model_dir"]
        custom_save_dir = self.inference_config["save_preds_dir"]

        if not save_to_train_save_dir and custom_save_dir:
            self.save_dir = custom_save_dir
        else:
            self.save_dir = train_save_dir

        now = (
            str(datetime.datetime.now())
            .replace(" ", "_")
            .replace(":", "_")
            .replace("-", "_")[:-10]
        )
        self.save_folder = os.path.join(self.save_dir, f"inference_results_{now}")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def predict_image(self, input_image, model=None):
        """
        Alias for predict_large_image if 2.5D; 2.5D torch model is xy generalizable

        :param numpy.ndarray input_image: image for prediction
        :param torch.nn.module model: trained model
        """
        return self.predict_large_image(input_image, model=model)

    def predict_large_image(self, input_image, model=None):
        """
        Runs prediction on entire image field of view. xy size is configurable, but it must be
        a power of 2. Input must be either 4 or 5 dimensions, and output is returned with the
        same dimensionality as given in input.

        Params:
        :param numpy.ndarray/torch.Tensor input_image: input image or image stack on which
                                                        to run prediction
        :param Torch.nn.Module model: trained model to use for prediction
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

        pred = model(img_tensor)
        return pred.detach().cpu().numpy()

    def run_inference(self):
        """
        Performs inference on the entire validation dataset.

        Model outputs are normalized and compared with ground truth through metrics specified
        in the metrics parameter in the inference section of the config.

        Metrics along with figures of both raw outputs, ground truth, and comparison overlays
        are saved in the tensorboard output at the specified save directory.
        """
        assert self.current_dataloader, "Select dataloader prior to inference"

        # init io and saving
        start = time.time()
        self.writer = SummaryWriter(log_dir=self.save_folder)
        self.model.eval()

        print("Running inference: \n")
        for current, sample in enumerate(self.current_dataloader):

            # pretty printing
            io_utils.show_progress_bar(self.current_dataloader, current)

            # get sample and target (removing single batch dimension)
            input_ = sample[0][0].cuda(device=self.device).float()
            target_ = sample[1][0].cuda(device=self.device).float()

            # run through model
            prediction = self.model(input_, validate_input=True)

            self.current_dataloader.dataset
