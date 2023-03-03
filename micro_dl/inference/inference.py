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
import microDL.micro_dl.utils.cli_utils as cli_utils
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.normalize as normalize
import micro_dl.utils.io_utils as io_utils
import micro_dl.inference.evaluation_metrics as inference_metrics


class TorchPredictor:
    """
    TorchPredictor object handles all procedures involved with model inference.
    The trainer uses a the model.py and dataset.py utility modules to instantiate and load a model
    and validation data for inference using a gunpowder backend.

    Params:
    :param torch_config
    """

    def __init__(self, torch_config, device=None) -> None:
        self.torch_config = torch_config

        self.zarr_dir = self.torch_config["zarr_dir"]
        self.network_config = self.torch_config["model"]
        self.training_config = self.torch_config["training"]
        self.dataset_config = self.torch_config["dataset"]
        self.inference_config = self.torch_config["inference"]
        self.preprocessing_config = self.torch_config["preprocessing"]

        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None
        self.current_dataloader = None
        self.inference_metrics = {}

        self.model = None
        if device:
            self.device = device
        else:
            self.device = self.inference_config["device"]

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
            inference_config=self.inference_config,
            network_config=self.network_config,
            dataset_config=self.dataset_config,
            data_split=self.get_inference_data_split(),
            device=self.device,
            workers=workers,
        )
        self.record_data_split(torch_data_container.data_split)
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
            dataset=ds.DatasetEnsemble(torch_datasets=train_dataset_list), shuffle=False
        )
        self.test_dataloader = DataLoader(
            dataset=ds.DatasetEnsemble(torch_datasets=test_dataset_list), shuffle=False
        )
        self.val_dataloader = DataLoader(
            dataset=ds.DatasetEnsemble(torch_datasets=val_dataset_list), shuffle=False
        )

    def get_inference_data_split(self):
        """
        Extracts the data split from the model directory referenced in inference. Data
        split is stored as a .yml local to the model. If the data split is not provided,
        returns false.

        Data split files be overriden by manually inputting the data split positions into
        the inference config section in the config file.

        :return dict data_split: dictionary of data split containing integer list of positions
                            OR decimal fractions indicating split under {'train', 'test', 'val'}
                            keys
        """
        if "custom_data_split" in self.inference_config:
            print("Using custom data split found in inference config.")
            return self.inference_config["custom_data_split"]

        model_dir = os.path.dirname(self.inference_config["model_dir"])
        data_split_file = os.path.join(model_dir, "data_splits.yml")

        if os.path.exists(data_split_file):
            print("Using saved data split found in model directory.")
            data_splits = aux_utils.read_config(data_split_file)
            timestamps = list(data_splits.keys())
            timestamps.sort(reverse=True)
            most_recent_split = timestamps[0]

            data_split = data_splits[most_recent_split]
            return data_split
        else:
            raise ValueError(
                f"No data_splits.yml file found in dir {model_dir}.\n"
                "Generate and save a data split file or override by providing"
                " a custom_data_split in inference config."
            )

    def record_data_split(self, data_split):
        """
        Records the given data split ('train', 'test', 'val') positions in the inference save
        folder with the predictions and metrics. Intention is to keep track of what data splits
        were used for what inference, in the event of a custom data split.

        :param dict data_split: dictionary of data split containing integer list of positions
                                    under {'train', 'test', 'val'} keys.
        """
        data_split_file = os.path.join(self.save_folder, "data_splits.yml")
        data_splits = {aux_utils.get_timestamp(): data_split}

        aux_utils.write_yaml(data_splits, data_split_file)

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

    def _get_source_information(self, dataloader, index):
        """
        Gets and returns the source information (position, path location, normalization
        metadata) needed to record for inference about a given position slice stack
        from the index dataset that contains it in the larger dataloader.

        Recall that the dataloader here must be built from a ConcatDataset dataset, which
        concatenates a series of InferenceDataset objects, each which has access to a
        single source node, access non-randomly a single zxy stack.

        :param Dataloader dataloader: dataloader built with respect to above docstring
        :param int index: index in the dataloader's dataset of the source dataset

        :return zarr.heirarchy position_group: zarr reader object of position
        :return str position_path: local path to source position within zarr store
        :return dict normalization_meta: normalization metadata from this position
        :return tuple spatial_window: offset and size of spatial window
        """
        current_torch_dataset = dataloader.dataset.datasets[index]
        source_node = current_torch_dataset.data_source[0]
        path_in_zarr_store = source_node.datasets[current_torch_dataset.active_key]

        # spatial window from dataset
        spatial_window = (
            current_torch_dataset.window_offset,
            current_torch_dataset.window_size,
        )

        # well position from source node
        position_path = pathlib.Path(path_in_zarr_store).parent
        if str(position_path)[0] == "/":
            position_path = str(position_path)[1:]
        else:
            position_path = str(position_path)
        position_path_global = os.path.join(source_node.filename, position_path)
        position_group = zarr.open(position_path_global, mode="r")

        # normalization meta
        normalization_meta = position_group.attrs.asdict()["normalization"]

        return position_group, position_path, normalization_meta, spatial_window

    def _unzscore(self, data, normalization_meta):
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

        un-zscores the data based upon the metadata

        :param np.ndarray data: 3d un-normalized input data
        :param dict data_statistics: dictionary of statistics containing precomputed norm
                                    values for dataset and FOV

        :return np.ndarray unzscored_data: denormed data of input data's shape and type
        """
        norm_type = self.dataset_config["normalization"]["type"]
        norm_scheme = self.dataset_config["normalization"]["scheme"]
        if norm_scheme == "FOV":
            statistics = normalization_meta["fov_statistics"]
        elif norm_scheme == "dataset":
            statistics = normalization_meta["dataset_statistics"]
        else:
            return data

        if norm_type == "median_and_iqr":
            unzscored_data = normalize.unzscore(
                data,
                zscore_median=statistics["median"],
                zscore_iqr=statistics["iqr"],
            )
        elif norm_type == "mean_and_std":
            unzscored_data = normalize.unzscore(
                data,
                zscore_median=statistics["mean"],
                zscore_iqr=statistics["std"],
            )

        return unzscored_data

    def _collapse_metrics_dict(self, metrics_dict):
        """
        Collapses metrics dict in the form of
            {metric_name: {index: metric,...}}
        to the form
            {metric_name: np.ndarray[metric1, metrics2,...]}

        :param dict metrics_dict: dict of metrics in the first format

        :return dict collapsed_metrics_dict: dict of metrics in the second format
        """
        collapsed_metrics_dict = {}
        for metric_name in metrics_dict:
            val_dict = metrics_dict[metric_name]
            values = [val_dict[index] for index in val_dict]
            collapsed_metrics_dict[metric_name] = np.array(values)

        return collapsed_metrics_dict

    def _get_metrics(
        self,
        target,
        prediction,
        metrics_list,
        metrics_orientations,
        path="unspecified",
        window=None,
    ):
        """
        Gets metrics for this target_/prediction pair in all the specified orientations for all the
        specified metrics.

        :param np.ndarray target: 5d target array (on cpu)
        :param np.ndarray prediction: 5d prediction array (on cpu)
        :param list metrics_list: list of strings indicating the name of a desired metric, for options
                                    see inference.evaluation_metrics. MetricsEstimator docstring
        :param list metrics_orientations: list of strings indicating the orientation to compute, for
                                    options see inference.evaluation_metrics. MetricsEstimator docstring
        :param tuple window: spatial window of this target/prediction pair in the larger arrays they
                                    come from.

        :return dict prediction_metrics: dict mapping orientation -> pd.dataframe of metrics for that
                                    orientation
        """
        metrics_estimator = inference_metrics.MetricsEstimator(metrics_list)
        prediction_metrics = {}

        # transpose target and prediction to be in xyz format
        # NOTE: This expects target and pred to be in the format bczyx!
        target = np.transpose(target, (0, 1, -2, -1, -3))
        prediction = np.transpose(prediction, (0, 1, -2, -1, -3))

        zstart, zend = window[0][0], window[0][0] + window[1][0]  # end = start + length
        pred_name = f"slice_{zstart}-{zend}"

        if "xy" in metrics_orientations:
            metrics_estimator.estimate_xy_metrics(
                target=target,
                prediction=prediction,
                pred_name=pred_name,
            )
            metrics_xy = self._collapse_metrics_dict(
                metrics_estimator.get_metrics_xy().to_dict()
            )
            prediction_metrics["xy"] = metrics_xy

        if "xyz" in metrics_orientations:
            metrics_estimator.estimate_xyz_metrics(
                target=target,
                prediction=prediction,
                pred_name=pred_name,
            )
            metrics_xyz = self._collapse_metrics_dict(
                metrics_estimator.get_metrics_xyz().to_dict()
            )
            prediction_metrics["xyz"] = metrics_xyz

        if "xz" in metrics_orientations:
            metrics_estimator.estimate_xz_metrics(
                target=target,
                prediction=prediction,
                pred_name=pred_name,
            )
            metrics_xz = self._collapse_metrics_dict(
                metrics_estimator.get_metrics_xz().to_dict()
            )
            prediction_metrics["xz"] = metrics_xz

        if "yz" in metrics_orientations:
            metrics_estimator.estimate_yz_metrics(
                target=target,
                prediction=prediction,
                pred_name=pred_name,
            )
            metrics_yz = self._collapse_metrics_dict(
                metrics_estimator.get_metrics_yz().to_dict()
            )
            prediction_metrics["yz"] = metrics_yz

        # format metrics
        tag = path + f"_{window}"
        self.inference_metrics[tag] = prediction_metrics

        return prediction_metrics

    def record_metrics(self, sample_information):
        """
        Handles metric recording in tensorboard.

        Metrics are saved position by position. If multiple scalar metric values are stored for a
        particular metric in a particular position, they are plotted along the axis they are calculated
        on.

        :param list sample_information: list of tuples containing information about each sample
                                in the form (position_group, position_path, normalization_meta, window)
        """
        for info_tuple in sample_information:
            _, position_path, normalization_meta, window = info_tuple
            position = position_path.split("/")[-1]
            sample_metrics = self.inference_metrics[position_path + f"_{window}"]

            for orientation in sample_metrics:
                scalar_dict = sample_metrics[orientation]
                pred_name = scalar_dict.pop("pred_name")[0]

                # generate a unique plot & tag for each orientation
                main_tag = f"{position}/{orientation}_{pred_name}"

                # Need to plot a line if metrics calculated along an axis
                if scalar_dict[list(scalar_dict.keys())[0]].shape[0] == 1:
                    self.writer.add_scalars(
                        main_tag=main_tag,
                        tag_scalar_dict=scalar_dict,
                    )
                else:
                    axis_length = scalar_dict[list(scalar_dict.keys())[0]].shape[0]
                    for i in range(axis_length):
                        scalar_dict_i = {}
                        for key in scalar_dict.keys():
                            scalar_dict_i[key] = scalar_dict[key][i]
                        self.writer.add_scalars(
                            main_tag=main_tag,
                            tag_scalar_dict=scalar_dict_i,
                            global_step=i,
                        )

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

        sample_information = []
        print("Running inference: \n")
        for current, sample in enumerate(self.current_dataloader):

            # pretty printing
            cli_utils.show_progress_bar(
                self.current_dataloader, current, process="predicting"
            )

            # get sample and target (removing single batch dimension)
            input_ = sample[0][0].cuda(device=self.device).float()
            target_ = sample[1][0].cuda(device=self.device).float()
            prediction = self.model(input_, validate_input=True)

            # pulling tensors off gpu prevents cuda errors
            input_ = input_.detach().cpu().numpy()
            target_ = target_.detach().cpu().numpy()
            prediction = prediction.detach().cpu().numpy()
            inference_data = {
                "input": input_,
                "target": target_,
                "pred": prediction,
            }

            # retrieve information about this sample
            information = self._get_source_information(self.current_dataloader, current)
            sample_information.append(information)
            position_group, position_path, normalization_meta, window = information

            # calculate metrics
            cli_utils.show_progress_bar(
                self.current_dataloader, current, process="computing metrics"
            )
            metrics_list = self.inference_config["metrics"]["metrics"]
            metrics_orientations = self.inference_config["metrics"]["orientations"]

            prediction_metrics = self._get_metrics(
                target=inference_data["target"],
                prediction=inference_data["pred"],
                metrics_list=metrics_list,
                metrics_orientations=metrics_orientations,
                path=position_path,
                window=window,
            )

            # unzscore our predictions/inputs/targets
            cli_utils.show_progress_bar(
                self.current_dataloader, current, process="recording figures"
            )
            modifier = io_utils.HCSZarrModifier(zarr_file=self.zarr_dir)

            assert len(input_.shape) == len(
                target_.shape
            ), "input, target, and prediction must have the same dimensionality"
            assert len(target_.shape) == len(
                prediction.shape
            ), "input, target, and prediction must have the same dimensionality"

            # not every channel is guaranteed to be normalized
            unzscore_channels = self.preprocessing_config["normalize"]["channel_ids"]
            if unzscore_channels == -1:
                unzscore_channels = list(range(modifier.channels))

            for key in inference_data:
                batch_data = inference_data[key]
                denormed_data = np.empty(batch_data.shape, dtype=batch_data.dtype)
                for batch_idx in range(batch_data.shape[0]):
                    for channel_idx in range(batch_data.shape[1]):
                        data = batch_data[batch_idx, channel_idx]

                        if channel_idx in unzscore_channels:
                            channel_name = modifier.channel_names[channel_idx]
                            data = self._unzscore(
                                data, normalization_meta[channel_name]
                            )
                        denormed_data[batch_idx, channel_idx] = data
                inference_data[key] = denormed_data

            # TODO: Add output to tiff file of inference_data

            # record predictions
            for data_name in inference_data:
                batch_data = inference_data[data_name]
                for batch_idx in range(batch_data.shape[0]):
                    for channel_idx in range(batch_data.shape[1]):
                        channel_name = modifier.channel_names[channel_idx]

                        img_data_stack = batch_data[batch_idx, channel_idx]
                        img_data = img_data_stack[img_data_stack.shape[0] // 2]

                        position = position_path.split("/")[-1]
                        self.writer.add_images(
                            tag=f"{position}/{data_name} {channel_name}: {window}",
                            img_tensor=torch.tensor(img_data),
                            dataformats="hw",
                        )

        # record metrics
        # self.writer.add_graph(self.model, input_)
        self.record_metrics(sample_information)

        self.writer.close()
        print(f"Done! Time taken: {time.time() - start:.2f}s")
        print(f"Predictions and metrics saved to {self.save_folder}")
