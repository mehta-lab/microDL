#%%
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
        Sets up testing environment
        """
        # generate configuration data
        self.temp = "temp_dir"
        self.train_config = {
            "data_dir": self.temp,
            "array_name": "arr_0",
            # "augmentations": None TODO implement
            "batch_size": 5,
            "split_ratio": {
                "train": 0.66,
                "test": 0.17,
                "val": 0.17,
            },
        }
        self.network_config = {"architecture": "2.5D", "debug_mode": False}
        self.dataset_config = {
            "target_channels": [1],
            "input_channels": [0],
            "window_size": (2, 256, 256),
        }
        self.num_channels = len(self.dataset_config["input_channels"]) + len(
            self.dataset_config["target_channels"]
        )

        # build zarr store accordingly
        self.build_zarr_store(self.temp, num_stores=6)

    def TakeDown(self):
        """
        Cleans up testing environment
        """
        # clean up zarr store
        shutil.rmtree(self.temp)

    def build_zarr_store(self, temp, num_stores):
        """
        Builds a test zarr store conforming to OME-NGFF Zarr format with 5d arrays
        in the directory 'temp'

        :param str temp: dir path to build zarr store in
        :param str zarr_name: name of zarr store inside temp dir (discluding extension)
        :param int num_stores: of zarr_stores to build
        :raises FileExistsError: cannot overwrite a currently written directory, so
                                temp must be a new directory
        """
        try:
            os.makedirs(temp, exist_ok=True)
        except Exception as e:
            raise FileExistsError(f"parent directory cannot already exist {e.args}")

        self.groups = []
        for i in range(num_stores):
            store = zarr.DirectoryStore(os.path.join(temp, f"example_{i}.zarr"))
            g1 = zarr.group(store=store, overwrite=True)
            self.groups.append(g1)

            arr_value = [0]
            arr_channels = self.num_channels
            arr_spatial = self.dataset_config["window_size"]

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
                            shape=(
                                [1, arr_channels] + [dim * 2 for dim in arr_spatial]
                            ),
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

            recurse_helper(g1, ["Row", "Col", "Pos", "arr"], 3, 3)

    def _test_basic_functionality(self):
        self.SetUp()

        try:
            torch_container = dataset_utils.TorchDatasetContainer(
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
                            sample[0], torch.tensor
                        ), "Samples produced are not tensors"
                        assert isinstance(
                            sample[1], torch.tensor
                        ), "Samples produced are not tensors"

                        # ensure sample is correct shape
                        window_size = self.dataset_config["window_size"]
                        batch_size = self.train_config["batch_size"]
                        num_target_channels = len(
                            self.dataset_config["target_channels"]
                        )
                        num_input_channels = len(self.dataset_config["input_channels"])

                        expected_target_size = tuple(
                            [batch_size, num_target_channels] + list(window_size)
                        )
                        assert sample[1].shape == expected_target_size(
                            f"Target samples produced of incorrect"
                            f"shape: expected: {expected_target_size} actual: {sample.shape}"
                        )

                        expected_input_size = tuple(
                            [batch_size, num_input_channels] + list(window_size)
                        )
                        assert sample[0].shape == expected_input_size(
                            f"Input samples produced of incorrect"
                            f"shape: expected: {expected_input_size} actual: {sample.shape}"
                        )

                except Exception as e:
                    raise AssertionError(
                        f"Error in loading samples from dataloader: {e.args}"
                    )

        self.TakeDown()


# %%
tester = TestDataset()
tester._test_basic_functionality()
