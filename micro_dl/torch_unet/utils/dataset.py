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


class TorchDataset(Dataset):
    """
    Based off of torch.utils.data.Dataset:
        - https://pytorch.org/docs/stable/data.html

    Custom dataset class that builds gunpowder pipelines constitution of multi-zarr source nodes
    and a series of augmentation nodes. This object will call from the gunpowder pipeline directly,
    and transform resulting data into tensors to be placed onto the gpu for processing.

    Multiprocessing is supported with num_workers > 0. However, there are non-fatal warnings about
    "...processes being terminated before shared CUDA tensors are released..." with torch 1.10.

    These warnings are discussed on the following post, and I believe have been since fixed:
        - https://github.com/pytorch/pytorch/issues/71187
    """

    def __init__(self, train_config, network_config):
        # TODO I still think that the container/dataset duality structure is superior. alter to conform
        """
        Inits an object which builds a testing, training, and validation dataset
        from .zarr data files using gunpowder pipelines

        :param str zarr_source_dir: directory of .zarr files containing all data
        :param list augmentation_nodes: list of gp nodes (type gp.BatchFilter) with which to build
                                        the augmentation pipeline
        """
        self.train_config = train_config
        self.network_config = network_config

        # acquire sources from the zarr directory
        array_spec = gp_utils.generate_array_spec(network_config)
        (
            self.train_source,
            self.test_source,
            self.val_source,
        ) = gp_utils.multi_zarr_source(
            zarr_dir=self.train_config["data_dir"],
            array_name=self.train_config["array_name"],
            array_spec=array_spec,
            data_split=self.train_config["split_ratio"],
        )

        # build augmentation nodes
        aug_nodes = []
        if "augmentations" in train_config:
            aug_nodes = gp_utils.generate_augmentation_nodes(
                train_config["augmentations"]
            )

        # build pipelines
        self.train_pipeline = self.build_pipeline(self.train_source, aug_nodes)
        self.test_pipeline = self.build_pipeline(self.test_source)
        self.val_pipeline = self.build_pipeline(self.val_source)

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

    def __len__(self):
        # TODO implement
        """
        Returns number of source data arrays in this dataset.
        """

    def __getitem__(self, idx):
        # TODO implement
        """
        If acting as a dataset object, returns the standard batch_request
        of this dataset, with random position and provider.

        If acting as a dataset container object, returns subsidary dataset
        objects.

        :param int idx: index of dataset item to transform and return
        """

        # if acting as dataset object
        if self.tf_dataset:
            sample = self.tf_dataset[idx]
            sample_input = sample[0]
            sample_target = sample[1]

            # match num dims as safety check
            samp_dims, targ_dims = len(sample_input.shape), len(sample_target.shape)
            for i in range(max(0, samp_dims - targ_dims)):
                sample_target = np.expand_dims(sample_target, 1)
            for i in range(max(0, targ_dims - samp_dims)):
                sample_input = np.expand_dims(sample_input, 1)

            if self.transforms:
                for transform in self.transforms:
                    sample_input = transform(sample_input)

            if self.target_transforms:
                for transform in self.target_transforms:
                    sample_target = transform(sample_target)

            return self.unpack(sample_input, sample_target)

        # if acting as container object of dataset objects
        else:
            keys = {}
            if self.val_dataset:
                keys["val"] = self.val_dataset
            if self.train_dataset:
                keys["train"] = self.train_dataset
            if self.test_dataset:
                keys["test"] = self.test_dataset

            if idx in keys:
                return keys[idx]
            else:
                raise KeyError(
                    f"This object is a container. Acceptable keys:{[k for k in keys]}"
                )

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
