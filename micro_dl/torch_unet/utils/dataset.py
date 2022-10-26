from socket import TIPC_DEST_DROPPABLE
import cv2
import collections
import gunpowder as gp
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.masks as mask_utils
import micro_dl.utils.normalize as norm_utils
import micro_dl.utils.train_utils as train_utils
import micro_dl.input.gunpowder_nodes as custom_nodes
import micro_dl.utils.gunpowder_utils as gp_utils


class TorchDatasetContainer(object):
    """
    Dataset container object which initalizes multiple TorchDatasets depending upon the parameters
    given in the training and network config files during initialization.

    Randomly selects samples to divy up sources between train, test, and validation datasets.
    """

    def __init__(
        self,
        train_config,
        network_config,
    ):
        # TODO I still think that the container/dataset duality structure is superior. alter to conform
        """
        Inits an object which builds a testing, training, and validation dataset
        from .zarr data files using gunpowder pipelines

        If acting as a container object:
        :param str zarr_source_dir: directory of .zarr files containing all data, by default None
        :param list augmentation_nodes: list of gp nodes (type gp.BatchFilter) with which to build
                                        the augmentation pipeline, by default None
        """
        self.train_config = train_config
        self.network_config = network_config

        # acquire sources from the zarr directory
        array_spec = gp_utils.generate_array_spec(network_config)
        (
            self.train_source,
            self.test_source,
            self.val_source,
            self.dataset_keys,
        ) = gp_utils.multi_zarr_source(
            zarr_dir=self.train_config["data_dir"],
            array_name=self.train_config["array_name"],
            array_spec=array_spec,
            data_split=self.train_config["split_ratio"],
        )
        try:
            assert len(self.test_source) > 0
            assert len(self.train_source) > 0
            assert len(self.val_source) > 0
        except Exception as e:
            raise AssertionError(
                f"All datasets must have at least one source node,"
                "not enough source arrays found: {e}"
            )

        # build augmentation nodes
        aug_nodes = []
        if "augmentations" in train_config:
            aug_nodes = gp_utils.generate_augmentation_nodes(
                train_config["augmentations"]
            )

        # assign each source subset to a child dataset object
        train_pipeline = self.build_pipeline(self.train_source, aug_nodes)
        test_pipeline = self.build_pipeline(self.test_source)
        val_pipeline = self.build_pipeline(self.val_source)

        self.train_dataset = TorchDataset(
            train_pipeline, self.train_source, self.dataset_keys
        )
        self.test_dataset = TorchDataset(
            test_pipeline, self.test_source, self.dataset_keys
        )
        self.val_dataset = TorchDataset(
            val_pipeline, self.val_source, self.dataset_keys
        )

    def build_pipeline(self, source, nodes=[]):
        """
        Builds a gunpowder data pipeline given a source node and a list of augmentation
        nodes

        :param gp.ZarrSource or tuple source: source node/tuple of source nodes from which to draw
        :param list nodes: list of augmentation nodes, by default empty list
        """
        # if source is multi_zarr_source, attach a RandomProvider
        if isinstance(source, tuple):
            source = [source, gp.RandomProvider()]

        # attach permanent nodes
        source = source + [gp.RandomLocation()]
        source = source + [custom_nodes.Normalize()]
        source = source + [custom_nodes.FlatFieldCorrect()]

        # attach additional nodes, if any, and sum
        pipeline = source + nodes
        pipeline = gp_utils.gpsum(pipeline, verbose=self.network_config["debug_mode"])

        return pipeline

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
        data_pipeline,
        data_source,
        data_keys,
        batch_size,
        target_channel_idx,
        input_channel_idx,
        spatial_window_size,
    ):
        """
        Creates a dataset object which draws samples directly from a gunpowder pipeline.

        :param gp.Pipeline data_pipeline: build data pipeline for given dataset task
        :param tuple data_source: tuple of gp.Source nodes which the given pipeline draws samples
        :param list data_keys: list of gp.ArrayKey objects which access the given source
        :param int batch_size: number of samples per batch
        :param tuple(int) target_channel_idx: indices of target channel(s) within zarr store
        :param tuple(int) input_channel_idx: indices of input channel(s) within zarr store
        :param tuple spatial_window_size: tuple of sample dimensions
                                    2D network, expects 2D tuple; dimensions yx
                                    2.5D network, expects 3D tuple; dimensions zyx
        """
        self.data_pipeline = data_pipeline
        self.data_source = data_source
        self.data_keys = data_keys
        self.batch_size = batch_size
        self.target_idx = target_channel_idx
        self.input_idx = input_channel_idx

        # safety checks: iterate through keys and data sources to ensure that they match
        voxel_size = None
        for key in self.data_keys:
            for i, source in enumerate(self.data_source):
                try:
                    assert len(source.array_specs) == len(
                        self.data_keys
                    ), f"number of keys {len(self.data_keys)} does not match number"
                    " of array specs {len(source.array_specs)}"

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
                        f"Error matching keys to source in dataset: {e}"
                    )

        # generate batch request depending on pipeline voxel size and input dims/idxs
        assert len(spatial_window_size) == len(
            voxel_size
        ), f"Incompatible voxel size {voxel_size}. "
        f"Must be equal to spatial_window_size {spatial_window_size}"

        batch_request = gp.BatchRequest()
        for key in self.data_keys:
            batch_request[key] = gp.Roi((0, 0, 0), spatial_window_size)
            # NOTE: the keymapping we are performing here makes it so that if
            # we DO end up generating multiple arrays at the group level
            # (for example one with and without flatfield correction), we can
            # access all of them by requesting that key. The index we request
            # in __getitem__ ends up being the index of our key.
        self.batch_request = batch_request

    def __len__(self):
        """
        Returns number of source data arrays in this dataset.
        """
        return len(self.data_source)

    def __getitem__(self, idx):
        # TODO this implementation might not play nice with our dataloader, since its call
        # signature messes with the key index. Watch out for that.
        """
        Requests a batch from the data pipeline by selecting the key from self.data_keys at
        index 'idx', and using that key to call a batch from its corresponding source using
        self.batch_request.

        :param int idx: index of the key you wish to use to access a random sample. Generally:
                        0 -> key for arr_0 in Pos_x
                        1 -> key for arr_1 in Pos_x
                                .
                                .
                                .
        """
        for i in range(self.batch_size):
            pass  # TODO implement

    def unpack(self, sample_input, sample_target):
        # TODO this may not be necessary any more
        """
        Helper function for unpacking tuples returned by some transformation objects
        (e.g. GenerateMasks) into outputs.

        Unpacking before returning allows transformation functions which produce variable amounts of
        additional tensor information to pack that information in tuples with the sample and target
        tensors.

        :param torch.tensor/tuple(torch.tensor) sample_input: input sample to unpack
        :param torch.tensor/tuple(torch.tensor) sample_target: target sample to unpack
        """
        inp, targ = type(sample_input), type(sample_target)

        if inp == list or inp == tuple:
            if targ == list or targ == tuple:
                return (*sample_input, *sample_target)
            else:
                return (*sample_input, sample_target)
        else:
            if targ == list or targ == tuple:
                return (sample_input, *sample_target)
            else:
                return (sample_input, sample_target)


class ToTensor(object):
    """
    Transformation. Converts input to torch.Tensor and returns. By default also places tensor
    on gpu.

    :param torch.device device: device transport tensor to
    """

    def __init__(self, device=torch.device("cuda")):
        self.device = device

    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            sample = sample.to(self.device)
        else:
            sample = torch.tensor(sample, dtype=torch.float32).to(self.device)
        return sample


class Resize(object):
    """
    NOTE: this function is currently unused (and already superceded). I wrote this to provide
    options for transforms performed after the dataloader. These should be removedas they will
    be superceded by gunpowder.

    Transformation. Resises called sample to 'output_size'.
    """

    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size

    def __call__(self, sample):
        sample = cv2.resize(sample, self.output_size)
        sample = cv2.resize(sample, self.output_size)
        return sample


class RandTile(object):
    """
    NOTE: this function is currently unused (and already superceded). I wrote this to provide
    options for transforms performed after the dataloader. These should be removedas they will
    be superceded by gunpowder.

    Transformation. Selects and returns random tile size 'tile_size' from input.
    """

    def __init__(self, tile_size=(256, 256), input_format="zxy"):
        Warning("RandTile is unrecommended for preprocessed data")
        self.tile_size = tile_size
        self.input_format = input_format

    def __call__(self, sample):
        if self.input_format == "zxy":
            x_ind, y_ind = -2, -1
        elif self.input_format == "xyz":
            x_ind, y_ind = -3, -2

        x_shape, y_shape = sample.shape[x_ind], sample.shape[y_ind]
        assert (
            self.tile_size[0] < y_shape and self.tile_size[1] < x_shape
        ), f"Sample size {(x_shape, y_shape)} must be greater than tile size {self.tile_size}."

        randx = np.random.randint(0, x_shape - self.tile_size[1])
        randy = np.random.randint(0, y_shape - self.tile_size[0])

        sample = sample[
            randy : randy + self.tile_size[0], randx : randx + self.tile_size[1]
        ]
        return sample


class RandFlip(object):
    """
    NOTE: this function is currently unused (and already superceded). I wrote this to provide
    options for transforms performed after the dataloader. These should be removedas they will
    be superceded by gunpowder.
    Transformation. Flips input in random direction and returns.
    """

    def __call__(self, sample):
        rand = np.random.randint(0, 2, 2)
        if rand[0] == 1:
            sample = np.flipud(sample)
        if rand[1] == 1:
            sample = np.fliplr(sample)
        return sample


class GenerateMasks(object):
    """
    NOTE: this function is currently unused (and already superceded). I wrote this to provide
    options for transforms performed after the dataloader. These should be removedas they will
    be superceded by gunpowder.

    Appends target channel thresholding based masks for each sample to the sample in a third
    channel, ordered respective to the order of each sample within its minibatch.

    Masks generated are torch tensors.

    :param str masking_type: type of thresholding to apply:
                                1.) Rosin/unimodal: https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf
                                2.) Otsu: https://en.wikipedia.org/wiki/Otsu%27s_method
    :param bool clipping: whether or not to clip the extraneous values in the data before
                                    thresholding
    :param int/tiple clip_amount: amount to clip from both ends of brightness histogram
                                    as a percentage (%) if clipping==True but clip_amount == 0,
                                    clip for default amount (2%)
    """

    def __init__(self, masking_type="rosin", clipping=False, clip_amount=0):

        assert masking_type in {"rosin", "unimodal", "otsu"}, (
            f"Unaccepted masking" "type: {masking_type}"
        )
        self.masking_type = masking_type
        self.clipping = clipping
        self.clip_amount = clip_amount

    def __call__(self, sample):
        original_sample = sample

        # convert to numpy
        if type(sample) != type(np.asarray([1, 1])):
            sample = sample.detach().cpu().numpy()

        # clip top and bottom 2% of images for better thresholding
        if self.clipping:
            if type(self.clip_amount) == tuple:
                sample = norm_utils.hist_clipping(
                    sample, self.clip_amount[0], 100 - self.clip_amount[1]
                )
            else:
                if self.clip_amount != 0:
                    sample = norm_utils.hist_clipping(
                        sample, self.clip_amount, 100 - self.clip_amount
                    )
                else:
                    sample = norm_utils.hist_clipping(sample)

        # generate masks
        masks = []
        for i in range(sample.shape[0]):
            if self.masking_type == "otsu":
                masks.append(mask_utils.create_otsu_mask(sample[i, 0, 0]))
            elif self.masking_type == "rosin" or self.masking_type == "unimodal":
                masks.append(mask_utils.create_unimodal_mask(sample[i, 0, 0]))
            else:
                raise NotImplementedError(
                    f"Masking type {self.masking_type} not implemented."
                )
                break
        masks = ToTensor()(np.asarray(masks)).unsqueeze(1).unsqueeze(1)

        return [original_sample, masks]


class Normalize(object):
    """
    NOTE: this function is currently unused. I wrote this to provide options
    for transforms performed after the dataloader. These should be removed
    as they will be superceded by gunpowder.

    Normalizes the sample sample according to the mode in init.

    Params:
    :param str mode: type of normalization to apply
            - one: normalizes sample values proportionally between 0 and 1
            - zeromax: centers sample around zero according to half of its
                        normalized (between -1 and 1) maximum
            - median: centers samples around zero, according to their respective
                        median, then normalizes (between -1 and 1)
            - mean: centers samples around zero, according to their respective
                        means, then normalizes (between -1 and 1)
    """

    def __init__(self, mode="max"):
        self.mode = mode

    def __call__(self, sample, scaling=1):
        """
        Forward call of Normalize
        Params:
            - sample -> torch.Tensor or numpy.ndarray: sample to normalize
            - scaling -> float: value to scale output normalization by
        """
        # determine module
        if isinstance(sample, torch.Tensor):
            module = torch
        elif isinstance(sample, np.ndarray):
            module = np
        else:
            raise NotImplementedError(
                "Only numpy array and torch tensor inputs handled."
            )

        # apply normalization
        if self.mode == "one":
            sample = (sample - module.min(sample)) / (
                module.max(sample) - module.min(sample)
            )
        elif self.mode == "zeromax":
            sample = (sample - module.min(sample)) / (
                module.max(sample) - module.min(sample)
            )
            sample = sample - module.max(sample) / 2
        elif self.mode == "median":
            sample = sample - module.median(sample)
            sample = sample / module.max(module.abs(sample))
        elif self.mode == "mean":
            sample = sample - module.mean(sample)
            sample = sample / module.max(module.abs(sample))
        else:
            raise NotImplementedError(f"Unhandled mode type: '{self.mode}'.")

        return sample * scaling


class RandomNoise(object):
    """
    NOTE: this function is currently unused. I wrote this to provide options
    for transforms performed after the dataloader. These should be removed
    as they will be superceded by gunpowder.

    Augmentation for applying random noise. High variance.

    :param str noise_type: type of noise to apply: 'gauss', 's&p', 'poisson', 'speckle'
    :param numpy.ndarray/torch.tensor sample: input sample
    :return numpy.ndarray/torch.tensor: noisy sample of type input type
    """

    def __init__(self, noise_type):
        self.noise_type = noise_type

    def __call__(self, sample):
        pt = False
        if isinstance(sample, torch.Tensor):
            pt = True
            sample = sample.detach().cpu().numpy()

        if self.noise_type == "gauss":
            row, col, ch = sample.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = sample + gauss
            return noisy

        elif self.noise_type == "s&p":
            row, col, ch = sample.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(sample)

            # Salt mode
            num_salt = np.ceil(amount * sample.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in sample.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * sample.size * (1.0 - s_vs_p))
            coords = [
                np.random.randint(0, i - 1, int(num_pepper)) for i in sample.shape
            ]
            out[coords] = 0
            return out

        elif self.noise_typ == "poisson":
            vals = len(np.unique(sample))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(sample * vals) / float(vals)
            return noisy

        elif self.noise_typ == "speckle":
            row, col, ch = sample.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = sample + sample * gauss
            return noisy

        if pt:
            sample = ToTensor()(sample)
        return sample
