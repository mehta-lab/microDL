import cv2
import collections
import gunpowder as gp
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

import micro_dl.utils.normalize as norm_utils
import micro_dl.input.gunpowder_nodes as custom_nodes
import micro_dl.utils.gunpowder_utils as gp_utils


class TorchDatasetContainer(object):
    """
    Dataset container object which initalizes multiple TorchDatasets depending upon the parameters
    given in the training and network config files during initialization.

    Randomly selects samples to divy up sources between train, test, and validation datasets.
    """

    def __init__(
        self, train_config, network_config, dataset_config, device=None, workers=0
    ):
        # TODO make necessary changes to configs to match structure implied in this module
        """
        Inits an object which builds a testing, training, and validation dataset
        from .zarr data files using gunpowder pipelines

        If acting as a container object:
        :param dict train_config: dict object of train_config
        :param dict train_config: dict object of train_config
        :param dict train_config: dict object of train_config
        :param str device: device on which to place tensors in child datasets,
                            by default, places on 'cuda'
        :param int workers: number of cpu workers for simultaneous data fetching
        """
        self.train_config = train_config
        self.network_config = network_config
        self.dataset_config = dataset_config
        self.device = device
        self.workers = workers

        # acquire sources from the zarr directory
        array_spec = gp_utils.generate_array_spec(network_config)
        (
            self.train_source,
            self.test_source,
            self.val_source,
            self.dataset_keys,
        ) = gp_utils.multi_zarr_source(
            zarr_dir=self.dataset_config["data_dir"],  # TODO config change
            array_name=self.dataset_config["array_name"],  # TODO config change
            array_spec=array_spec,
            data_split=self.dataset_config["split_ratio"],  # TODO config change
        )
        try:
            assert len(self.test_source) > 0
            assert len(self.train_source) > 0
            assert len(self.val_source) > 0
        except Exception as e:
            raise AssertionError(
                f"All datasets must have at least one source node / zarr store,"
                f" not enough source arrays found.\n {e.args}"
            )

        # build augmentation nodes
        aug_nodes = []
        if "augmentations" in train_config:
            aug_nodes = gp_utils.generate_augmentation_nodes(
                train_config["augmentations"]  # TODO config change
            )

        # assign each source subset to a child dataset object
        self.train_dataset = self.init_torch_dataset(self.train_source, aug_nodes)
        self.test_dataset = self.init_torch_dataset(self.test_source, aug_nodes)
        self.val_dataset = self.init_torch_dataset(self.val_source, aug_nodes)

    def init_torch_dataset(self, source, augmentation_nodes):
        """
        Initializes a torch dataset to sample 'source' through the given
        augmentations 'augmentations'.

        :param tuple(gp.ZarrSource) source: tuple of source nodes representing the
                                            dataset sample space
        :param list augmentation_nodes: list of augmentation nodes in order
        """
        # NOTE: not passing the whole dataset config is a design choice here. The
        # elements of the config are a black box until theyre indexed. I do this
        # to make them more readable. This can change with PyDantic later
        dataset = TorchDataset(
            data_source=source,
            augmentation_nodes=augmentation_nodes,
            data_keys=self.dataset_keys,
            batch_size=self.dataset_config["batch_size"],  # TODO config change
            target_channel_idx=tuple(
                self.dataset_config["target_channels"]
            ),  # TODO config change
            input_channel_idx=tuple(
                self.dataset_config["input_channels"]
            ),  # TODO config change
            spatial_window_size=tuple(
                self.dataset_config["window_size"]
            ),  # TODO config change
            device=self.device,
            workers=self.workers,
        )
        return dataset

    def __getitem__(self, idx):
        """
        Provides indexing capabilities to reference train, test, and val datasets

        :param int or str idx: index/key of dataset to retrieve:
                                train -> 0 or 'train'
                                test -> 1 or 'test'
                                val -> 2 or 'val'
        """
        if isinstance(idx, str):
            return {
                "train": self.train_dataset,
                "test": self.test_dataset,
                "val": self.val_dataset,
            }[idx]
        else:
            return [self.train_dataset, self.test_dataset, self.val_dataset][idx]


class TorchDataset(Dataset):
    """
    Based off of torch.utils.data.Dataset:
        - https://pytorch.org/docs/stable/data.html

    Custom dataset class that builds gunpowder pipelines composed of multi-zarr source nodes
    and a series of augmentation nodes. This object will call from the gunpowder pipeline directly,
    and transform resulting data into tensors to be placed onto the gpu for processing.

    Multiprocessing is supported with num_workers > 0. However, there are non-fatal warnings about
    "...processes being terminated before shared CUDA tensors are released..." with torch 1.10.

    These warnings are discussed on the following post, and I believe have been since fixed:
        - https://github.com/pytorch/pytorch/issues/71187
    """

    def __init__(
        self,
        data_source,
        augmentation_nodes,
        data_keys,
        batch_size,
        target_channel_idx,
        input_channel_idx,
        spatial_window_size,
        device,
        workers,
    ):
        """
        Creates a dataset object which draws samples directly from a gunpowder pipeline.

        :param tuple(gp.ZarrSource) source: tuple of source nodes representing the
                                            dataset sample space
        :param list augmentation_nodes: list of augmentation nodes in order
        :param tuple data_source: tuple of gp.Source nodes which the given pipeline draws samples
        :param list data_keys: list of gp.ArrayKey objects which access the given source
        :param int batch_size: number of samples per batch
        :param tuple(int) target_channel_idx: indices of target channel(s) within zarr store
        :param tuple(int) input_channel_idx: indices of input channel(s) within zarr store
        :param tuple spatial_window_size: tuple of sample dimensions, specifies batch request ROI
                                    2D network, expects 2D tuple; dimensions yx
                                    2.5D network, expects 3D tuple; dimensions zyx
        :param str device: device on which to place tensors in child datasets,
                            by default, places on 'cuda'
        :param int workers: number of simultaneous threads reading data into batch requests
        """
        self.data_source = data_source
        self.augmentation_nodes = augmentation_nodes
        self.data_keys = data_keys
        self.batch_size = batch_size
        self.target_idx = target_channel_idx
        self.input_idx = input_channel_idx
        self.window_size = spatial_window_size
        self.device = device
        self.active_key = 0
        self.workers = workers - 1

        # safety checks: iterate through keys and data sources to ensure that they match
        voxel_size = None
        for key in self.data_keys:
            for i, source in enumerate(self.data_source):
                try:
                    assert len(source.array_specs) == len(
                        self.data_keys
                    ), f"number of keys {len(self.data_keys)} does not match number"
                    f" of array specs {len(source.array_specs)}"

                    # check that all voxel sizes are the same
                    array_spec = source.array_specs[key]
                    if not voxel_size:
                        voxel_size = array_spec.voxel_size
                    else:
                        assert (
                            voxel_size == array_spec.voxel_size
                        ), f"Voxel size of array {array_spec.voxel_size} does not match"
                        f" voxel size of previous array {voxel_size}."
                except Exception as e:
                    raise AssertionError(
                        f"Error matching keys to source in dataset: {e.args}"
                    )

        # generate batch request depending on pipeline voxel size and input dims/idxs
        assert len(self.window_size) == len(voxel_size), (
            f"Incompatible voxel size {voxel_size}. "
            f"Must be same length as spatial_window_size {self.window_size}"
        )

        batch_request = gp.BatchRequest()
        for key in self.data_keys:
            batch_request[key] = gp.Roi((0,) * len(self.window_size), self.window_size)
            # NOTE: the keymapping we are performing here makes it so that if
            # we DO end up generating multiple arrays at the group level
            # (for example one with and without flatfield correction), we can
            # access all of them by requesting that key. The index we request
            # in __getitem__ ends up being the index of our key.
        self.batch_request = batch_request

        # build our pipeline and the generator structure that uses it
        self.pipeline = self._build_pipeline()
        self.batch_generator = self._generate_batches()

    def __len__(self):
        """
        Returns number of source data arrays in this dataset.
        """
        return len(self.data_source)

    def __getitem__(self, idx=0):
        """
        Requests a batch from the data pipeline using the key selected by self.use_key,
        and applying that key to call a batch from its corresponding source using
        self.batch_request.

        :param int idx: index of the key you wish to use to access a random sample. Generally:
                        0 -> key for arr_0 in Pos_x
                        1 -> key for arr_1 in Pos_x
                                .
                                .
                                .
        """
        assert self.active_key != None, "No active key. Try '.use_key()'"

        # TODO: this implementation pulls 5 x 256 x 256 of all channels. We may not want
        # all of those slices in all channels if theyre note being used. Fix this inefficiency

        sample = next(self.batch_generator)
        sample_data = sample[self.active_key].data

        # NOTE We assume the .zarr ALWAYS has an extra batch channel.
        # SO, 3d -> 5d data, 2d -> 4d data

        # remove extra dimension from stack node
        sample_data = sample_data[:, 0, ...]

        # stack multiple channels
        full_input = []
        for idx in self.input_idx:  # assumes bczyx or bcyx
            channel_input = sample_data[:, idx - 1, ...]
            full_input.append(channel_input)
        full_input = np.stack(full_input, 1)

        full_target = []
        for idx in self.target_idx:  # assumes bczyx or bcyx
            channel_target = sample_data[:, idx, ...]
            full_target.append(channel_target)
        full_target = np.stack(full_target, 1)

        if len(full_target.shape) == 5:
            # target is always 2 dimensional, we select middle z dim
            middle_z_idx = full_target.shape[-3] // 2
            full_target = np.expand_dims(full_target[..., middle_z_idx, :, :], -3)

        # convert to tensor and place onto gpu
        convert = ToTensor(self.device)
        input_, target_ = convert(full_input), convert(full_target)

        return (input_, target_)

    def use_key(self, selection):
        """
        Sets self.active_key to selection if selection is an ArrayKey. If selection is an int,
        sets self.active_key to the index in self.data_keys given by selection.

        :param int or gp.ArrayKey selection: key index of key in self.data_keys to activate
        """
        if isinstance(selection, int):
            try:
                self.active_key = self.data_keys[selection]
            except IndexError as e:
                raise IndexError(
                    f"Handling exception {e.args}: index of selection"
                    " must be within length of data_keys"
                )
        elif isinstance(selection, gp.ArrayKey):
            # TODO change this assertion to reference the source-> arrayspec-> keys
            assert (
                selection in self.data_keys
            ), "Given key not associated with dataset sources"
            self.active_key = selection
        else:
            raise AttributeError("Selection must be int or gp.ArrayKey")

    def _build_pipeline(self):
        """
        Builds a gunpowder data pipeline given a source node and a list of augmentation
        nodes.

        :param gp.ZarrSource or tuple source: source node/tuple of source nodes from which to draw
        :param list nodes: list of augmentation nodes, by default empty list

        :return gp.Pipeline pipeline: see name
        """
        # if source is multi_zarr_source, attach a RandomProvider
        source = self.data_source
        if isinstance(source, tuple):
            source = [source, gp.RandomProvider()]

        # attach permanent nodes
        source = source + [gp.RandomLocation()]

        # TODO implement
        # source = source + [custom_nodes.Normalize()]
        # source = source + [custom_nodes.FlatFieldCorrect()]

        batch_creation = []
        batch_creation.append(
            gp.PreCache(cache_size=150, num_workers=max(0, self.workers))
        )  # TODO make these parameters variable
        batch_creation.append(gp.Stack(self.batch_size))

        # attach additional nodes, if any, and sum
        pipeline = source + self.augmentation_nodes + batch_creation

        pipeline = gp_utils.gpsum(pipeline, verbose=False)

        return pipeline

    def _generate_batches(self):
        """
        Returns pipeline as a generator. This is done in a separate method from __getitem__()
        to preserve compatibility with the torch dataloader's item calling signature
        while also performing appropriate context management for the pipeline via generation.
        See:
            https://github.com/funkey/gunpowder/issues/181

        :param gp.Pipeline pipeline: pipeline to generate batches from
        :param gp.BatchRequest request: batch request for pipeline

        :yield gp.Batch batch: single batch yielded from pipeline at each generation
        :rtype: Iterator[gp.Batch]
        """
        with gp.build(self.pipeline):
            while True:
                yield self.pipeline.request_batch(self.batch_request)


class ToTensor(object):
    """
    Transformation. Converts input to torch.Tensor and returns. By default also places tensor
    on gpu.

    :param torch.device device: device transport tensor to
    """

    def __init__(self, device=torch.device("cuda")):
        self.device = device

    def __call__(self, sample):
        """
        Perform transformation.

        :param torch.tensor or numpy.ndarray sample: data to convert to tensor and place on device
        :return torch.tensor sample: converted data on device
        """
        if isinstance(sample, torch.Tensor):
            sample = sample.to(self.device)
        else:
            sample = torch.tensor(sample, dtype=torch.float32).to(self.device)
        return sample
