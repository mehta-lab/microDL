import glob
import gunpowder as gp
import os
import pathlib
import random
import re
import zarr

from matplotlib import test

import micro_dl.input.gunpowder_nodes as nodes
import micro_dl.utils.augmentation_utils as aug_utils
import micro_dl.utils.io_utils as io_utils


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
            pipeline += nodes.LogNode(str(prefix), time_nodes=verbose)
            prefix += 1
    return pipeline


def build_sources(zarr_store_dir, store_well_paths, arr_spec):
    """
    Builds a source for every well_path (position) in a zarr store, specified by
    store_well_paths and zarr_dir, and returns each source.

    The sources will have a different key for each array type at each well.
    For example, if your wells contain:
        |- arr_0
        |- arr_1
        .
        .
        |- arr_n

    This method will build a source for each well, each of which can be accessed by a
    any of a list corresponding of gp.ArrayKey keys that are returned in order:

        [keys_0, keys_1, ... , key_n]

    The keys used to access each store map to the corresponding array type.

    Note: the implication with this implementation is that all wells contain the same
    array types (and number of array types). This should not be used for non-uniform
    numbers of array types across a single store.

    :param str zarr_store_dir: path to zarr directory to build sources for
    :param collections.defaultdict store_well_paths: mapping of all well paths in zarr store
    :param gp.ArraySpec arrspec: ArraySpec pertaining to datasets (supports one global spec)

    :return list sources: dictionary of datasets locations and corresponding arraykeys
    :return list keys: list of ArrayKeys for each array type, shared across sources
    """

    sources, keys = [], []

    for path in list(store_well_paths):
        array_types = store_well_paths[path]

        path_keys = [gp.ArrayKey(ar_type) for ar_type in array_types]

        if len(keys) == 0:
            keys = path_keys
        else:
            identity = lambda x: x.identifier
            assert list(map(identity, path_keys)) == list(
                map(identity, keys)
            ), f"Found different array types for path {os.path.join(zarr_store_dir, path)}"

        dataset_dict = {}
        for i, key in enumerate(keys):
            dataset_dict[key] = os.path.join(path, array_types[i])

        spec_dict = {}
        for i, dataset_key in enumerate(keys):
            spec_dict[dataset_key] = arr_spec

        source = gp.ZarrSource(
            filename=zarr_store_dir, datasets=dataset_dict, array_specs=spec_dict
        )

        sources.append(source)

    return sources, keys


def get_zarr_source_position(zarr_source):
    """
    Gets the position that this zarr_source refers to

    For context, this requires that all datasets inside this zarr_source refer to the
    same position

    This method retrieves the position by scraping the path. It assumes that the
    postion name in the path is numbered, and that the number is the position:

        '/..plate_name../..well_name../Pos_<position_number_as_int>/...arr_name...'

    :param gp.ZarrSource zarr_source: source that refers to a single position in an
                                HCS compatible zarr store
    """
    zarr_dir = zarr_source.filename
    modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir, enable_creation=False)

    source_position = ""
    for key in zarr_source.datasets:
        path = zarr_source.datasets[key]

        position = list(map(int, re.findall(r"\d+", path)))[-2]
        if isinstance(source_position, str):
            source_position = position

        assert source_position == position, (
            "Found two datasets with different positions",
            f"in the same source: {position} and {source_position}",
        )
        source_position = position

    return source_position


def multi_zarr_source(
    zarr_dir, array_name="*", array_spec=None, data_split={}, use_recorded_split=False
):
    """
    Generates a tuple of source nodes for for each dataset type (train, test, val),
    containing one source node for every well in the zarr_dir specified.

    Applies same specification to all source datasets. Note that all source datasets of the
    same name exhibit **key sharing**. That is, the source key will be the same for all datasets
    of name 'arr_0' (for example) and a different source key will be shared amongst
    'arr_0_preprocessed' sources. This feature is only relevant if array_name matches
    multiple arrays in the specified zarr stores.

    Note: The zarr store in 'zarr_dir' must have the _same number of array types_. This is to
    enable key sharing, which is necessary for the RandomProvider node to be able to utilize all
    positions.

    Note: The group hierarchy of the zarr arrays must follow the OME-NGFF HCS zarr format. That
    is:
    |-Root_dir
         |-Row_0
              |-Col_0
                  |-Pos_0
                        |
                        |-arr_0
                        |-arr_1
                        .
                        .
                        |-arr_N
                  .
                  .
                  |-Pos_N
              .
              .
              |-Col_N
          .
          .
          |-Row_N

    Notice that the depth here is 3 group orderings followed by array names. This is crucial
    for this function to work properly.

    :param str zarr_dir: path to HCS-compatible zarr store.
    :param str array_name: name of the data container at bottom level of zarr tree,
                            by default, accesses all containers
    :param gp.ArraySpec array_spec: specification for zarr datasets, defaults to None
    :param dict data_split: dict containing fractions  to split data for train, test, validation.
                            Fields must be 'train', 'test', and 'val'. By default does not split
                            and returns one source tuple. Is overridden by "use_recorded_split".
    :param bool use_recorded_split: if true, will use recorded data split stored in top-level .zattrs
                            of "zarr_dir". by default is false

    :return tuple all_sources: (if no data_split) multi-source node from zarr_dir stores (equivalent to
                            s_1 + ... + s_n)
    :return tuple train_source: (if data_split) random subset of sources for training
    :return tuple test_source: (if data_split) random subset of sources for testing
    :return tuple val_source: (if data_split) random subset of sources for validation
    :return list all_keys: list of shared keys for each dataset type across all source subsets.
    """

    # generate the relative paths from each global parent directory
    zarr_files = [zarr_dir]

    zarr_stores = {}
    most_recent_array_types = {}
    most_recent_fname = ""
    for zarr_fname in zarr_files:
        zarr_array_paths = list(
            glob.glob(
                os.path.join(zarr_fname, os.path.join("*/*/*", array_name)),
                recursive=True,
            )
        )

        array_types = {}
        well_paths = {}

        for path in zarr_array_paths:
            array_type = os.path.basename(os.path.normpath(path))

            # TODO maybe a safer way to do this string creation.
            well_path = path.replace(zarr_fname, "").replace(array_type, "")

            # map every well to the types that well contains
            array_types[array_type] = None
            if well_path in well_paths:
                well_paths[well_path].append(array_type)
            else:
                well_paths[well_path] = [array_type]

        zarr_stores[zarr_fname] = well_paths

        # safety check: all zarr directories must contain the same base array types
        #               to allow for key sharing
        if len(most_recent_array_types) > 0:
            assert list(most_recent_array_types) == list(array_types), (
                f"Zarr store:\n\t {pathlib.Path(most_recent_fname).parts[-1]}"
                "\ncontains array types:"
                f"\n\t{list(zarr_stores[most_recent_fname])}"
                f"\nWhile Zarr store: \n\t {pathlib.Path(zarr_fname).parts[-1]}"
                "\ncontains array types"
                f"\n\t{list(array_types)}"
                "\n\nArray types of all stores must match to enable key sharing."
            )
        most_recent_array_types = array_types
        most_recent_fname = zarr_fname

    # build a source for *each position* (OME-NGFF well-level) in every given zarr store
    # with keys for each type of data stored at that position
    # https://ngff.openmicroscopy.org/latest/#well-md
    all_sources = []
    all_keys = []
    for zarr_fname in zarr_files:
        store_well_paths = zarr_stores[zarr_fname]

        store_sources, store_keys = build_sources(
            zarr_fname, store_well_paths, array_spec
        )
        all_sources.extend(store_sources)

        if len(all_keys) == 0:
            all_keys = store_keys
        else:
            identity = lambda x: x.identifier
            assert list(map(identity, store_keys)) == list(
                map(identity, all_keys)
            ), f"Found different array types in zarr store {zarr_fname}"

    if len(data_split) > 0 and use_recorded_split == False:
        assert "train" in data_split and "test" in data_split and "val" in data_split, (
            f"Incorrect format for data_split: {data_split}."
            " \n Must contain 'train', 'test', and 'val' "
        )

        # randomly generate split
        random.shuffle(all_sources)
        train_idx = int(len(all_sources) * data_split["train"])
        test_idx = int(len(all_sources) * (data_split["train"] + data_split["test"]))
        val_idx = len(all_sources)

        train_source = tuple(all_sources[0:train_idx])
        test_source = tuple(all_sources[train_idx:test_idx])
        val_source = tuple(all_sources[test_idx:val_idx])

        # record the positions of each source with their data split
        position_metadata = {}
        split = ["train", "test", "val"]
        for i, source_list in enumerate([train_source, test_source, val_source]):
            positions = list(map(get_zarr_source_position, source_list))
            position_metadata[split[i]] = positions
        plate_level_store = zarr.open(zarr_dir, mode="a")
        plate_level_store.attrs.update({"data_split_positions": position_metadata})

        return train_source, test_source, val_source, all_keys
    elif use_recorded_split:
        # read recorded split and validate
        plate_level_store = zarr.open(zarr_dir, mode="a")
        data_split = plate_level_store.attrs.asdict()["data_split_positions"]

        assert "train" in data_split and "test" in data_split and "val" in data_split, (
            f"Incorrect format for data_split: {data_split}."
            " \n Must contain 'train', 'test', and 'val' "
        )

        # map each source to its position
        train_source, test_source, val_source = [], [], []
        for i, source in enumerate(all_sources):
            source_position = get_zarr_source_position(source)
            if source_position in data_split["train"]:
                train_source.append(source)
            elif source_position in data_split["test"]:
                test_source.append(source)
            elif source_position in data_split["val"]:
                val_source.append(source)

        train_source = tuple(train_source)
        test_source = tuple(test_source)
        val_source = tuple(val_source)

        return train_source, test_source, val_source, all_keys
    else:
        source = tuple(all_sources)
        return source, all_keys


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


def generate_augmentation_nodes(aug_config, augmentation_keys):
    """
    Returns a list of augmentation nodes as specified in 'aug_config'.
    Return order is insensitive to the order of creation in augmentation_config...
    Augmentations can be given in any order, they will always be returned in a
    compatible sequence.

    :param augmentation_config: dict of augmentation type -> hyperparameters,
                                see torch_config readme.md for more details
    :return list aug_nodes: list of augmentation nodes in order
    """
    augmentation_builder = aug_utils.AugmentationNodeBuilder(
        aug_config,
        noise_key=augmentation_keys,
        blur_key=augmentation_keys,
        intensities_key=augmentation_keys,
        defect_key=augmentation_keys,
        shear_key=augmentation_keys,
    )
    augmentation_builder.build_nodes()
    aug_nodes = augmentation_builder.get_nodes()

    return aug_nodes
