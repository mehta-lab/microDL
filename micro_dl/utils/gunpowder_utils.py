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
            dataset_dict[key] = array_types[i]

        spec_dict = {}
        for i, dataset_key in enumerate(keys):
            spec_dict[dataset_key] = arr_spec

        source = gp.ZarrSource(
            filename=os.path.join(zarr_store_dir, path),
            datasets=dataset_dict,
            array_specs=spec_dict,
        )

        sources.append(source)

    return sources, keys


def multi_zarr_source(zarr_dir, array_name="*", array_spec=None, data_split=None):
    """
    Generates a tuple of source nodes for for each dataset type (train, test, val),
    containing one source node for every well in the zarr_dir specified.
    Applies same specification to all source datasets.

    Applies same specification to all source datasets. Note that all source datasets of the
    same name exhibit **key sharing**. That is, the source key will be the same for all datasets
    of name 'arr_0' (for example) and a different source key will be shared amongst
    'arr_0_preprocessed' sources. This feature is only relevant if array_name matches
    multiple arrays in the specified zarr stores.

    Note: The zarr stores in 'zarr_dir' must have the _same number of array types_. This is to
    enable key sharing, which is necessary for the RandomProvider node to be able to utilize them.

    Note: The group hierarchy of the zarr arrays must follow the OME-NGFF zarr format. That
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

    :param str zarr_dir: path to zarr directory.
    :param str array_name: name of the data container at bottom level of zarr tree,
                            by default, accesses all containers
    :param gp.ArraySpec array_spec: specification for zarr datasets, defaults to None
    :param dict data_split: dict containing fractions to split data for train, test, validation,
                            fields must be 'train', 'test', and 'val',
                            by default does not split and returns one source tuple

    :return tuple all_sources: (if no data_split) multi-source node from zarr_dir stores (equivalent to
                            s_1 + ... + s_n)
    :return tuple train_source: (if data_split) random subset of sources for training
    :return tuple test_source: (if data_split) random subset of sources for testing
    :return tuple val_source: (if data_split) random subset of sources for validation
    :return list all_keys: list of shared keys for each dataset type across all source subsets.
    """

    # generate the relative paths from each global parent directory
    zarr_files = [
        os.path.join(zarr_dir, fname)
        for fname in os.listdir(zarr_dir)
        if pathlib.Path(fname).suffix == ".zarr"
    ]

    zarr_stores = {}
    # collections.defaultdict(lambda: collections.defaultdict(lambda: []))
    most_recent_array_types = {}
    most_recent_fname = ""
    for zarr_fname in zarr_files:
        zarr_array_paths = list(
            glob.glob(
                os.path.join(zarr_fname, os.path.join("*/*/*", array_name)),
                recursive=True,
            )
        )

        array_types = {}  # collections.defaultdict(lambda: [])
        well_paths = {}  # collections.defaultdict(lambda: [])

        for path in zarr_array_paths:
            array_type = os.path.basename(os.path.normpath(path))

            # TODO maybe a safer way to do this string creation.
            well_path = path.replace(zarr_fname + "/", "").replace(array_type, "")

            # map every well to the types that well contains
            array_types[array_type] = None
            if well_path in well_paths:
                well_paths[well_path].extend(list(array_types))
            else:
                well_paths[well_path] = list(array_types)

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

    if len(data_split) > 0:
        assert "train" in data_split and "test" in data_split, (
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


def generate_augmentation_nodes(aug_config):
    # TODO Change this to be insensitive of ordering and automatically find the
    #     compatible sequence
    """Returns a list of augmentation nodes as specified in 'augmentation_config'.
    Return order is sensitive to the order of creation in augmentation_config...
    Augmentations should always be given in a sequentially compatible ordering.

    :param augmentation_config: dict of augmentation type -> hyperparameters,
                                see torch_config readme.md for more details
    :return list aug_nodes: list of augmentation nodes in order
    """
    # generate copy to avoid mutation issues
    aug_config = aug_config.copy()

    aug_nodes = []
    for aug_name in aug_config:
        parameters = aug_config[aug_name]
        aug_nodes.append(init_aug_node(aug_name, parameters))

    return aug_nodes


def init_aug_node(aug_name, parameters):
    """
    Acts as a general initatialization method, which takes a augmentation name and
    parameters and initializes and returns a gunpowder node corresponding to that
    augmentation

    :param str aug_name: name of augmentation
    :param dict parameters: dict of parameter names and values for augmentation

    :return gp.BatchFilter aug_node: single gunpowder node for augmentation
    """
    # TODO Will need a discussion about what augmentations to support
    if aug_name == "Defect":
        pass
    elif aug_name == "ElasticDeform":
        pass
    elif aug_name == "RandomIntensity":
        pass
    elif aug_name == "Noise":
        pass
    elif aug_name == "SimpleGeometric":
        pass
    elif aug_name == "":
        pass
    elif aug_name == "":
        pass
    elif aug_name == "":
        pass
    elif aug_name == "":
        pass
