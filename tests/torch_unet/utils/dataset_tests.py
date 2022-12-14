# %%
import collections
import glob
import shutil
import torch
from torch.utils.data import DataLoader
import numpy as np
import itertools
import os
import unittest
import zarr

import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")

import micro_dl.torch_unet.utils.dataset as dataset_utils
import micro_dl.torch_unet.utils.io as io_utils


class TestDataset(unittest.TestCase):
    def SetUp(self):
        """
        Initialize different test configurations to run tests on
        """
        # generate configuration data
        self.temp = "temp_dir"
        self.augmentations = {
            "transpose": {
                "transpose_only": [1, 2],
            },
            "mirror": {
                "mirror_only": [1, 2],
            },
            "rotate": {
                "rotation_interval": [0, 3.14],
            },
            "zoom": {
                "scale_interval": [0.7, 1.3],
            },
            "blur": {
                "mode": "gaussian",
                "width_range": [3, 7],
                "sigma": 0.1,
                "prob": 0.2,
                "blur_channels": [0],
            },
            "shear": {
                "angle_range": [0.31574673, 0.31574674],
                "prob": 0.2,
                "shear_middle_slice_only": [1, 2],
            },
            "intensity_jitter": {
                "mode": "gaussian",
                "scale_range": [0.7, 1.3],
                "shift_range": [-0.15, 0.15],
                "norm_before_shift": True,
                "jitter_demeaned": True,
                "prob": 0.2,
            },
            "noise": {
                "mode": "gaussian",
                "seed": 14156,
                "clip": True,
                "prob": 0.2,
                "noise_channels": [0],
            },
        }
        self.train_config_batch = {
            "augmentations": self.augmentations,
            "batch_size": 5,
            "split_ratio": {
                "train": 0.66,
                "test": 0.17,
                "val": 0.17,
            },
            "samples_per_epoch": 30,
        }
        self.train_config_single = {
            "augmentations": self.augmentations,
            "batch_size": 1,
            "split_ratio": {
                "train": 0.66,
                "test": 0.17,
                "val": 0.17,
            },
            "samples_per_epoch": 30,
        }
        self.all_train_configs = [self.train_config_batch, self.train_config_single]

        self.network_config_2d = {"architecture": "2D", "debug_mode": False}
        self.network_config_25d = {"architecture": "2.5D", "debug_mode": False}
        self.all_network_configs = [self.network_config_25d]  # self.network_config_2d,
        self.dataset_config_2d = {
            "array_name": "arr_0",
            "target_channels": [1],
            "input_channels": [0],
            "window_size": (256, 256),
            "normalization": {
                "scheme": "FOV",
                "type": "median_and_iqr",
            },
            "min_foreground_fraction": 0.2,
            "flatfield_correct": True,
            "batch_size": 16,
            "split_ratio": {
                "test": 0.15,
                "train": 0.7,
                "val": 0.15,
            },
        }
        self.dataset_config_25d = {
            "array_name": "arr_0",
            "target_channels": [1],
            "input_channels": [0, 2],
            "window_size": (3, 256, 256),
            "normalization": {
                "scheme": "FOV",
                "type": "median_and_iqr",
            },
            "min_foreground_fraction": 0.2,
            "flatfield_correct": True,
            "batch_size": 16,
            "split_ratio": {
                "test": 0.15,
                "train": 0.7,
                "val": 0.15,
            },
        }
        self.all_dataset_configs = [self.dataset_config_25d]  # self.dataset_config_2d,

    def tearDown(self):
        """
        Cleans up testing environment
        """
        # clean up zarr store
        if os.path.exists(self.temp):
            shutil.rmtree(self.temp)

    def build_zarr_store(self, temp, arr_spatial, num_stores=1):
        # TODO rewrite using io_utils.
        """
        Builds a test zarr store conforming to OME-NGFF Zarr format with 5d arrays
        in the directory 'temp'

        :param str temp: dir path to build zarr store in
        :param str zarr_name: name of zarr store inside temp dir (discluding extension)
        :param int num_stores: of zarr_stores to build
        :param tuple arr_spatial: spatial dimensions of data
        :raises FileExistsError: cannot overwrite a currently written directory, so
                                temp must be a new directory
        """
        try:
            os.makedirs(temp, exist_ok=True)
        except Exception as e:
            raise FileExistsError(f"parent directory cannot already exist {e.args}")

        def recurse_helper(group, names, subgroups, depth):
            """
            Recursively makes heirarchies of 'num_subgroups' subgroups until 'depth' is reached
            as children of 'group', filling the bottom most groups with arrays of value
            arr_value, and incrementing arr_value.

            :param zarr.heirarchy.Group group: Parent group ('.zarr' store)
            :param list names: names subgroups at each depth level + [name of arr]
            :param int subgroups: number of subgroups at each level (width)
            :param int depth: height of subgroup tree (height)
            :param int ar_channels: number of channels of data array
            :param int depth: window size of data array
            """
            if depth == 0:
                for j in range(subgroups):
                    z1 = zarr.open(
                        os.path.join(
                            group.store.dir_path(),
                            group.path,
                            names[-depth - 1] + f"_{j}",
                        ),
                        mode="w",
                        shape=([1, arr_channels] + [dim * 2 for dim in arr_spatial]),
                        chunks=([1, 1] + list(arr_spatial)),
                        dtype="float32",
                    )
                    val = arr_value.pop(0)
                    z1[:] = val
                    arr_value.append(val + 1)
            else:
                for j in range(subgroups):
                    subgroup = group.create_group(names[-depth - 1] + f"_{j}")
                    recurse_helper(subgroup, names, subgroups, depth - 1)

        # set parameters for store creation
        max_input_channels = 0
        max_target_channels = 0
        for config in self.all_dataset_configs:
            max_input_channels = max(len(config["input_channels"]), max_input_channels)
            max_target_channels = max(
                len(config["target_channels"]), max_target_channels
            )
        self.num_channels = max_input_channels + max_target_channels

        # build stores
        self.groups = []
        for i in range(num_stores):
            store_path = os.path.join(temp, f"example_{i}.zarr")
            store = zarr.DirectoryStore(store_path)
            g1 = zarr.group(store=store, overwrite=True)
            self.groups.append(g1)

            arr_value = [0]
            arr_channels = self.num_channels

            recurse_helper(g1, ["Row", "Col", "Pos", "arr"], 3, 3)
        self.zarr_dir = store_path

    def _test_basic_functionality(self):
        """
        Tests functionality with configuration described in self.SetUp().

        Pulls one sample from each dataset created in setup with every key
        corresponding to that dataset (see train_config.array_name)

        :raises AssertionError: Errors if errors found in initiation
        :raises AssertionError: Errors if setting key and accessing dataset produces
                                unexpected behavior
        :raises AssertionError: Errors if samples produced are not of expected size,
                                shape, and type
        """
        try:
            torch_container = dataset_utils.TorchDatasetContainer(
                self.zarr_dir,
                self.train_config,
                self.network_config,
                self.dataset_config,
                device=torch.device("cuda:0"),
            )
        except Exception as e:
            raise AssertionError(f"Failed dataset or container initiation: {e.args}")

        train_dataset = torch_container.train_dataset
        test_dataset = torch_container.test_dataset
        val_dataset = torch_container.val_dataset

        all_datasets = [train_dataset, test_dataset, val_dataset]

        for dataset in all_datasets:
            for key in dataset.data_keys:
                try:
                    dataset.use_key(key)
                except Exception as e:
                    raise AssertionError(f"Error in setting active key: {e.args}")
                try:
                    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                    for sample in dataloader:
                        # ensure sample is a tuple of tensors
                        assert (
                            len(sample) == 2
                        ), f"Target-input tuple shape :{len(sample)} not 2"
                        assert isinstance(
                            sample[0], torch.Tensor
                        ), "Samples produced are not tensors"
                        assert isinstance(
                            sample[1], torch.Tensor
                        ), "Samples produced are not tensors"

                        # ensure sample is correct shape
                        window_size = self.dataset_config["window_size"]
                        batch_size = self.train_config["batch_size"]
                        num_target_channels = len(
                            self.dataset_config["target_channels"]
                        )
                        num_input_channels = len(self.dataset_config["input_channels"])

                        # remove extra torch batch dim
                        input_ = sample[0][0]
                        target_ = sample[1][0]

                        expected_input_size = tuple(
                            [batch_size, num_input_channels] + list(window_size)
                        )
                        assert input_.shape == expected_input_size, (
                            f"Input samples produced of incorrect"
                            f" shape - expected: {expected_input_size} actual: {input_.shape}"
                        )

                        if len(window_size) == 3:
                            expected_target_size = tuple(
                                [batch_size, num_target_channels]
                                + [1]
                                + list(window_size)[1:]
                            )
                        else:
                            expected_target_size = tuple(
                                [batch_size, num_target_channels] + list(window_size)
                            )
                        assert target_.shape == expected_target_size, (
                            f"Target samples produced of incorrect "
                            f" shape - expected: {expected_target_size} actual: {target_.shape}"
                        )
                        break

                except Exception as e:
                    raise AssertionError(
                        f"Error in loading samples from dataloader: {e.args}"
                    )

    def _all_test_configurations(self, test):
        """
        Run specified test on all possible data sampling configurations. Pairs dataset
        and network configurations by index (necessary as there are exclusive parameters).
        With every pair of dataset and network configs, tries every possible training config.

        :param str test: test to run (must be attribute of self)
        """

        for i in range(len(self.all_network_configs)):
            self.dataset_config = self.all_dataset_configs[i]
            self.network_config = self.all_network_configs[i]

            self.build_zarr_store(
                temp=self.temp, arr_spatial=self.dataset_config["window_size"]
            )

            for train_config in self.all_train_configs:
                self.train_config = train_config

                # test functionality with each config
                try:
                    test()
                except Exception as e:
                    # self.tearDown()
                    raise Exception(
                        f"\n\n Exception caught with configuration:"
                        f"\n\n training: {self.train_config}"
                        f"\n\n dataset: {self.dataset_config}"
                        f"\n\n network: {self.network_config}"
                    )

            self.tearDown()

    # ------- tests --------#

    def test_functionality(self):
        """
        Test basic functionality on given configurations:
            - builds zarr store
            - tests container and dataset initation stability
            - tests that sample size, shape, and type matches expected
        """
        self.SetUp()
        self._all_test_configurations(self._test_basic_functionality)
        # self.tearDown()


# %%
tester = TestDataset()
# %%
tester.test_functionality()
# %%
