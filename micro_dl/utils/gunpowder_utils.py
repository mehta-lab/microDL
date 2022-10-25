import glob
import gunpowder as gp
import os
import pathlib
import random

from matplotlib import test

import micro_dl.input.gunpowder_nodes as nodes


def gpsum(nodelist, verbose=True):
    """
    Interleaves printing nodes in between nodes listed in nodelist.
    Returns pipeline of nodelist. If verbose set to true pipeline will print
    call sequence information upon each batch request.

    :param list nodelist: list of nodes to construct pipeline from
    :param bool verbose: whether to include gpprint notes, defaults to True
    :return gp.Node pipeline: gunpowder data pipeline
    """
    pipeline = nodelist.pop(0)
    prefix = 0
    while len(nodelist) > 0:
        pipeline += nodelist.pop(0)
        if verbose:
            pipeline += nodes.LogNode(str(prefix))
            prefix += 1
    return pipeline


def build_source_dicts(datasets, arrspec, array_keys):
    """
    Builds a dataset dict to allow a gp source node to access all arrnames.
    Primarily used as a helper function for the MultiZarrSource node

    :param list(str) relative_path: list of paths to data arrays inside zarr store
    :param gp.ArraySpec arrspec: ArraySpec pertaining to data (only one global spec supported)
    :param list arraykeys: list of shared keys to index into each dataset

    :return dict dataset_dict: dictionary of datasets locations and corresponding arraykeys
    :return dict spec_dict: dictionary of array specifications corresponding to dataset_dict

    """
    dataset_dict = {}
    for i, dataset_key in enumerate(array_keys):
        dataset_dict[dataset_key] = datasets[i]

    spec_dict = None
    if arrspec != None:
        spec_dict = {}
        for i, dataset_key in enumerate(array_keys):
            spec_dict[dataset_key] = arrspec

    return dataset_dict, spec_dict


def multi_zarr_source(zarr_dir, array_name="*", array_spec=None, data_split=None):
    """
    Generates a tuple of source nodes for for each dataset type (train, test, val),
    containing one source node for every one of the zarr stores in 'zarr_dir'.

    Applies same specification to all source datasets. Note that all source datasets of the
    same name exhibit **key sharing**. That is, the source key will be the same for all datasets
    of name 'arr_0' (for example) and a different source key will be shared amongst
    'arr_0_preprocessed' sources. This feature is only relevant if array_name matches
    multiple arrays in the specified zarr stores.

    Note: The zarr stores in 'zarr_dir' must have the _same number of arrays_. This is to allow
    key sharing, which is necessary for the RandomProvider node to be able to utilize them.

    :param str zarr_dir: path to parent directory of zarr stores.
    :param str array_name: name of the data container at bottom level of zarr tree,
                            by default, accesses all container types
    :param gp.ArraySpec array_spec: specification for zarr datasets, defaults to None
    :param dict data_split: dict containing fractions to split data for train, test, validation,
                            fields must be 'train', 'test', and 'val',
                            by default does not split and returns one source tuple

    :return tuple source: (if no data_split) multi-source node from zarr_dir stores (equivalent to s_1 + ... + s_n)
    :return tuple train_source: (if data_split) random subset of sources for training
    :return tuple test_source: (if data_split) random subset of sources for testing
    :return tuple val_source: (if data_split) random subset of sources for validation
    :return list dataset_keys: list of shared keys for each dataset type across all source subsets.
    """

    sources = []
    dataset_keys = []

    # generate the relative paths from each global parent directory
    zarr_files = [
        os.path.join(zarr_dir, fname)
        for fname in os.listdir(zarr_dir)
        if pathlib.Path(fname).suffix == ".zarr"
    ]
    assert len(zarr_files) > 0, f"Unable to find zarr dirs at: {zarr_dir}"

    for zarr_fname in zarr_files:
        global_paths = list(
            glob.glob(
                os.path.join(zarr_fname, os.path.join("*/*/*", array_name)),
                recursive=True,
            )
        )
        assert (
            len(global_paths) > 0
        ), f"Unable to find zarr data arrays at: {os.path.join(zarr_dir, zarr_fname)}"
        relative_paths = [
            glob_path.replace(os.path.join(zarr_dir, zarr_fname), "")
            for glob_path in global_paths
        ]

        # generate keys to share between all sources
        if len(dataset_keys) == 0:
            dataset_keys = [
                gp.ArrayKey(f"key_{i}") for i, arr in enumerate(relative_paths)
            ]
            assert (
                len(dataset_keys) > 0
            ), f"0 arrays found in zarr store at: {zarr_fname}"

        # create a source for each array
        dataset_dict, spec_dict = build_source_dicts(
            relative_paths, array_spec, dataset_keys
        )
        source = gp.ZarrSource(
            filename=os.path.join(zarr_dir, zarr_fname),
            datasets=dataset_dict,
            array_specs=spec_dict,
        )
        sources.append(source)

    if len(data_split) > 0:
        assert "train" in data_split and "test" in data_split, (
            f"Incorrect format for data_split: {data_split}."
            " \n Must contain 'train', 'test', and 'val' "
        )

        # randomly generate split
        random.shuffle(sources)
        train_idx = int(len(sources) * data_split["train"])
        test_idx = int(len(sources) * data_split["train"] + data_split["test"])
        val_idx = len(sources)

        train_source = tuple(sources[0:train_idx])
        test_source = tuple(sources[train_idx:test_idx])
        val_source = tuple(sources[test_idx:val_idx])
        return train_source, test_source, val_source, dataset_keys
    else:
        source = tuple(sources)
        return source, dataset_keys


def generate_array_spec(network_config):
    """
    Generates an array_spec for the zarr source data based upon the model used in
    training (as specified in network config)

    :param network_config: config dictionary containint
    :returns gp.ArraySpec array_spec: ArraySpec metadata compatible with given config
    """
    assert (
        "architecture" in network_config
    ), f"no 'architecture' specified in network config"

    if network_config["architecture"] == "2D":
        # 2D data will be 4D, 2 spatial dims
        voxel_size = (1, 1)
    elif network_config["architecture"] == "2.5D":
        # 2.5D data will be 5D, 2 spatial dims
        voxel_size = (1, 1, 1)
    else:
        raise AttributeError(
            f"Architecture {network_config['architecture']} not supported"
        )

    array_spec = gp.ArraySpec(
        interpolatable=True,
        voxel_size=voxel_size,
    )

    return array_spec
