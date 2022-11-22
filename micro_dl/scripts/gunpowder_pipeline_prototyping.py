# %%
import collections
import glob
from lib2to3.pgen2.token import NOTEQUAL
import pathlib
import numpy as np
import scipy.signal as signal
import cv2
from random import random
from pandas import array
import zarr
from multiprocessing.pool import Pool

import os
import gunpowder as gp

import matplotlib.pyplot as plt

import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")

from micro_dl.input.dataset import apply_affine_transform
from micro_dl.utils.normalize import unzscore
from micro_dl.utils.gunpowder_utils import gpsum, multi_zarr_source
from micro_dl.input.gunpowder_nodes import (
    IntensityAugment,
    BlurAugment,
    ShearAugment,
    LogNode,
)

# %%
# ------------------------------- #
#      set up data & helpers      #
# ------------------------------- #


def gpsum(nodelist, track_nodes=False, time_nodes=False):
    """
    Interleaves printing nodes in between nodes listed in nodelist.
    Returns pipeline of nodelist. If verbose set to true pipeline will print
    call sequence information upon each batch request.

    :param list nodelist: list of nodes to construct pipeline from
    :param bool track_nodes: whether to include gpprint notes, defaults to False
    :param bool time_nodes: only valid if 'track_nodes'. times the request-receive
                            delay for this point in the pipeline
    :return gp.Node pipeline: gunpowder data pipeline
    """
    pipeline = nodelist.pop(0)
    prefix = 0
    while len(nodelist) > 0:
        pipeline += nodelist.pop(0)
        if track_nodes:
            pipeline += LogNode(str(prefix), time_nodes=time_nodes)
            prefix += 1
    return pipeline


def build_sources(zarr_dir, store_well_paths, arr_spec):
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

    :param str zarr_dir: path to zarr directory to build sources for
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
            ), f"Found different array types for path {os.path.join(zarr_dir, path)}"

        dataset_dict = {}
        for i, key in enumerate(keys):
            dataset_dict[key] = os.path.join(path, array_types[i])

        spec_dict = {}
        for i, dataset_key in enumerate(keys):
            spec_dict[dataset_key] = arr_spec

        source = gp.ZarrSource(
            filename=zarr_dir, datasets=dataset_dict, array_specs=spec_dict
        )

        sources.append(source)

    return sources, keys


def MultiZarrSource(zarr_dir, array_name="*", array_spec=None):
    """
    Generates a list of source nodes containing one source node for each of the zarr stores in
    'zarr_dir'. Applies same specification to all source datasets.

    Note: The zarr stores in 'zarr_dir' must have the _same number of array types_. This is to
    enable key sharing, which is necessary for the RandomProvider node to be able to utilize them.

    :param str zarr_dir: path to zarr directory.
    :param str array_name: name of the data container at bottom level of zarr tree,
                            by default, accesses all containers
    :param gp.ArraySpec array_spec: specification for zarr datasets, defaults to None

    :return gp.Node source: multi-source node from zarr_dir stores (equivelent to s_1 + ... + s_n)
    :return list keys: list of keys for each source.
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
            well_path = path.replace(zarr_fname, "").replace(array_type, "")

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
                f"zarr store {most_recent_fname} contains array types "
                f"{list(zarr_stores[most_recent_fname])} while zarr store {zarr_fname}"
                f"contains array types {list(array_types)}. Array types of all stores"
                "must match to enable key sharing."
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

    return tuple(all_sources), all_keys


# %%
# -------------------------------------------- #
# SOURCE + RAND PROV + RANDOM LOC + SIMPLE AUG #
#                                              #
#       According to William's Suggestions     #
# -------------------------------------------- #

zarr_dir = os.path.join(
    "/hpc/projects/CompMicro/projects/infected_cell_imaging/Image_preprocessing/"
    "Exp_2022_10_25_VeroCellNuclMemStain/VeroCell_40X_11NA",
)
# zarr_dir = "/tmp/VeroCell_40X_11NA"

print("Taking data from: ", zarr_dir)
print("building sources...", end="")
spec = gp.ArraySpec(interpolatable=True, voxel_size=gp.Coordinate((1, 1, 1)))
multi_source, keys = MultiZarrSource(zarr_dir, array_spec=spec)
raw = keys[0]
print("done")

print("building nodes...", end="")
random_location = gp.RandomLocation()
random_provider = gp.RandomProvider()
simple_aug = gp.SimpleAugment(
    transpose_only=(1, 2), mirror_only=(1, 2), mirror_probs=(0, 0, 0, 0, 0, 0, 0, 0)
)
elastic_aug = gp.ElasticAugment(
    control_point_spacing=(1, 1, 1),
    jitter_sigma=(0, 0, 0),
    rotation_interval=(0, 0),
    scale_interval=(1, 1),
    spatial_dims=2,
)
blur_aug = BlurAugment(
    array=raw,
    mode="gaussian",
    width_range=(23, 25),
    sigma=1,
    prob=0.5,
    blur_channels=(0,),
)
shear_aug = ShearAugment(
    array=raw,
    angle_range=(-30, 30),
    prob=0.33,
)
intensity_aug = IntensityAugment(
    array=raw,
    jitter_channels=(0,),
    jitter_demeaned=True,
    shift_range=(-0.15, 0.15),
    scale_range=(0.5, 1.5),
    norm_before_shift=True,
)

profiling = gp.PrintProfilingStats(every=1)

cache = gp.PreCache(cache_size=2500, num_workers=16)
batch_stack = gp.Stack(3)
print("done")
print("building pipeline...", end="")
batch_pipeline = gpsum(
    [
        multi_source[0],
        random_provider,
        random_location,
        simple_aug,
        elastic_aug,
        shear_aug,
        intensity_aug,
        blur_aug,
        # cache,  # important to cache upstream of stack
        batch_stack,
    ],
    track_nodes=False,
    time_nodes=False,
)

request = gp.BatchRequest()
request[raw] = gp.Roi((15, 728, 728), (5, 256, 256))

with gp.build(batch_pipeline) as pipeline:
    print("done")

    print("requesting batch...", end="")
    import time

    start = time.time()
    for i in range(1):
        sample = pipeline.request_batch(request=request)
        data = sample[raw].data[:, 0, ...]
    print("done")
    print(time.time() - start, " seconds")

    print("returned data shape:", data.shape)
    print(f"max: {np.max(data)}, min: {np.min(data)}")

    vmax = None
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for row in range(3):
        for channel in range(3):
            if channel == 0:
                ax[row][channel].imshow(np.mean(data[row][channel], 0), cmap="gray")
                ax[row][channel].set_title(
                    f"max:{np.max(np.mean(data[row][channel],0)):.3f}, "
                    f"min:{np.max(np.mean(data[row][channel],0)):.3f}"
                )
            else:
                ax[row][channel].imshow(data[row][channel][3])
                ax[row][channel].set_title(
                    f"max:{np.max(data[row][channel][3]):.3f}, "
                    f"min:{np.max(data[row][channel][3]):.3f}"
                )
    plt.show()
# %%
