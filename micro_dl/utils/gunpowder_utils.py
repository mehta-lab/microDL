import glob
import gunpowder as gp
import os
import pathlib

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


def MultiZarrSource(zarr_dir, array_name="*", array_spec=None):
    """
    Generates a list of source nodes containing one source node for each of the zarr stores in
    'zarr_dir'. Applies same specification to all source datasets.

    Note: The zarr stores in 'zarr_dir' must have the _same number of arrays_. This is to allow
    key sharing. If not, the RandomProvider node will not be able to utilize them.

    :param str zarr_dir: path to zarr directory.
    :param str array_name: name of the data container at bottom level of zarr tree,
                            by default, accesses all containers
    :param gp.ArraySpec array_spec: specification for zarr datasets, defaults to None

    :return gp.Node source: multi-source node from zarr_dir stores (equivelent to s_1 + ... + s_n)
    :return list keys: list of keys for each source.
    """

    sources = []
    source_keys = []

    # generate the relative paths from each global parent directory
    zarr_files = [
        os.path.join(zarr_dir, fname)
        for fname in os.listdir(zarr_dir)
        if pathlib.Path(fname).suffix == ".zarr"
    ]

    for zarr_fname in zarr_files:
        global_paths = list(
            glob.glob(
                os.path.join(zarr_fname, os.path.join("*/*/*", array_name)),
                recursive=True,
            )
        )
        relative_paths = [
            glob_path.replace(os.path.join(zarr_dir, zarr_fname), "")
            for glob_path in global_paths
        ]

        # generate keys to share between all sources
        if len(source_keys) == 0:
            source_keys = [
                gp.ArrayKey(f"key_{i}") for i, arr in enumerate(relative_paths)
            ]
            assert (
                len(source_keys) > 0
            ), f"0 arrays found in zarr store at: {zarr_fname}"

        # create a source for each array
        dataset_dict, spec_dict = build_source_dicts(
            relative_paths, array_spec, source_keys
        )
        source = gp.ZarrSource(
            filename=os.path.join(zarr_dir, zarr_fname),
            datasets=dataset_dict,
            array_specs=spec_dict,
        )
        sources.append(source)
    return tuple(sources), source_keys
