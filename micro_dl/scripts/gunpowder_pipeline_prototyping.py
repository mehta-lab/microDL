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

# %%
# ------------------------------- #
#      set up data & helpers      #
# ------------------------------- #


class LogNode(gp.BatchFilter):
    def __init__(self, prefix, log_image_dir=None, time_nodes=False):
        """Custom gunpowder node for printing data path

        :param str prefix: prefix to print before message (ex: node number)
        :param str log_image_dir: directory to log data to as it travels downstream
                                    by default, is None, and will not log data
        :param bool time_nodes: whether to time request-receive delay for this node
        """
        self.prefix = prefix
        self.log_image_dir = log_image_dir
        self.time_nodes = time_nodes

    def prepare(self, request):
        """Prints message during call to upstream request
        :param gp.BatchRequest request: current gp request
        """
        print(f"{self.prefix}\t Upstream provider: {self.upstream_providers[0]}")
        if self.time_nodes:
            self.time = time.time()

    def process(self, batch, request):
        """Prints message during call to downstream request
        :param gp.BatchRequest batch: batch returned by request from memory
        :param gp.BatchRequest request: current gp request
        """
        print(f"{self.prefix}\tBatch going downstream: {batch}")
        if self.log_image_dir:
            pass  # TODO implement this using the datalogging utils.
        if self.time_nodes:
            print(f"Took{time.time() - self.time:.2f}s to get back to me \n")


class ShearAugment(gp.BatchFilter):
    def __init__(
        self,
        array=None,
        angle_range=(0, 0),
        prob=0.1,
    ):
        """
        Custom gunpowder augmentation node for applying shear in xy.

        Assumes xy spatial dimensions of ROI are the last two dimensions. Shear is performed
        along x (cols) dimension. This is intended to pair with a random rotation performed by
        ElasticAugment, to achieve shearing along all possible axes.

        Note: Assumes channel dimensions is last non-voxel channel. Data must be in this format

        :param gp.ArrayKey array: key to array to perform blurring on. This is provided for
                                to enable shearing of one arraykey in the batch, but not
                                the other. If no key provided, applies to all key:data pairs
                                in request.
        :param tuple(float, float) angle_range: range of angles in degrees of shear. To prevent
                                            data corruption, angle must be within (0,30)
        :param float prob: probability of applying shear
        """
        assert (
            abs(angle_range[0]) <= 30 and abs(angle_range[1]) <= 30
        ), "bounds of shearing angle"
        f" range must be withing [-30, 30] but are {angle_range}"

        self.array_key = array
        self.angle_range = angle_range
        self.prob = prob

    def prepare(self, request: gp.BatchRequest):
        """
        Prepare request going upstream for data retrieval: increases request size
        in rows dimension to accommodate for information lost in corners during shearing.

        :param gp.BatchRequest request: current gp downstream request

        :return gp.BatchRequest request: modified or unmodified request depending on
                                        random roll > threshold of self.prob
        """
        if self.array_key == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_key
            if not isinstance(self.array_key, list):
                keys = [self.array_key]

        # determine if shearing
        self.apply_shear = np.random.uniform(0, 1) <= self.prob
        if self.apply_shear:
            # determine angle for this shear
            self.angle = np.random.uniform(*self.angle_range)
            new_request = gp.BatchRequest()
            expand_fraction = abs(self.angle / 90)

            # grow roi by extra pixels
            for key in keys:
                roi = request[key].roi
                self.extra_pixels = int(roi.get_shape()[-1] * expand_fraction)

                if roi.dims() == 2:
                    context = gp.Coordinate((0, self.extra_pixels))
                else:
                    length_dims = roi.dims() - 2
                    context = gp.Coordinate(
                        tuple([0] * length_dims + [0, self.extra_pixels])
                    )

                new_context_roi = roi.grow(context, context)
                new_request[key] = new_context_roi

            return new_request
        else:
            return request

    def process(self, batch: gp.Batch, request: gp.BatchRequest):
        """
        Blur batch going downstream for by convolution with kernel defined in init
        and crop to original size. Valid convolution always used.

        :param gp.BatchRequest request: current gp downstream request
        :param gp.Batch batch: current batch traveling downstream
        """
        if self.array_key == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_key
            if not isinstance(self.array_key, list):
                keys = [self.array_key]

        if self.apply_shear:
            for key in keys:
                batch_data = batch[key].data
                roi = request[key].roi

                output_shape = list(batch_data.shape[:-3]) + list(
                    request[key].roi.get_shape()
                )
                sheared_data = np.empty(output_shape, dtype=batch_data.dtype)

                # TODO: datasets with an additional index beyond channel and batch may
                #      break in loops. Can be safely implemented with dynamic recursion.
                for batch_idx in range(batch_data.shape[0]):
                    for channel_idx in range(batch_data.shape[1]):
                        data = batch_data[batch_idx, channel_idx]
                        if roi.dims() == 2:
                            data = np.expand_dims(data, 0)
                            data = apply_affine_transform(data, shear=self.angle)[0]
                        else:
                            data = apply_affine_transform(data, shear=self.angle)

                        if self.angle > 0:
                            data = data[..., :, : -self.extra_pixels * 2]
                        else:
                            data = data[..., :, self.extra_pixels * 2 :]

                        sheared_data[batch_idx, channel_idx] = data.astype(
                            batch_data.dtype
                        )

                batch[key] = batch[key].crop(request[key].roi)
                batch[key].data = sheared_data


class BlurAugment(gp.BatchFilter):
    def __init__(
        self,
        array=None,
        mode="gaussian",
        width_range=(1, 7),
        sigma=0.1,
        prob=0.2,
        blur_channels=None,
    ):
        """
        Custom gunpowder augmentation node for applying blur in xy.
        Assumes xy spatial dimensions of ROI are the last two dimensions.
        Implementation inspired by:
           https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html

        Note: Only symmetric blur kernels are supported
        Note: Assumes channel dimensions is last non-voxel channel. Data must be in this format

        :param gp.ArrayKey array: key to array to perform blurring on. This is provided for
                                blur to enable blurring of one arraykey in the batch, but not
                                the other. If no key provided, applies to all key:data pairs
                                in request.
        :param str mode: Type of blur (scipy documentation), defaults to "gaussian"
        :param float sigma: sigma of blur kernel
        :param tuple(int, int) width_range: range of pixel widths of blur kernel in xy
        :param float prob: probability of applying blur
        :param tuple(int) blur_channels: blur only these channel indices in channel dim 'channel_dim'
        """
        assert (
            width_range[0] % 2 == 1 and width_range[1] % 2 == 1
        ), "width range bounds must be odd"

        self.array_key = array
        self.mode = mode
        self.width_range = width_range
        self.prob = prob
        self.sigma = sigma
        self.blur_channels = blur_channels

        self._init_kernels()

    def prepare(self, request: gp.BatchRequest):
        """
        Prepare request going upstream for data retrieval by increasing request size
        to accommodate for kernel width. This is to ensure valid convolution of region.

        :param gp.BatchRequest request: current gp upstream request
        """
        if self.array_key == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_key
            if not isinstance(self.array_key, list):
                keys = [self.array_key]

        new_request = gp.BatchRequest()

        # set random kernel and get width
        index = np.random.randint(0, len(self.kernels))
        self.active_kernel = self.kernels[index]
        width = self.active_kernel.shape[0]

        for key in keys:
            # expand roi for each key to provide for needed context
            roi = request[key].roi
            assert roi.dims() > 1, "Must provide at least 2 spatial dims in ROI"

            if roi.dims() == 2:
                context = gp.Coordinate((width // 2, width // 2))
            else:
                width_dims = roi.dims() - 2
                context = gp.Coordinate(
                    tuple([0] * width_dims + [width // 2, width // 2])
                )

            new_context_roi = roi.grow(context, context)
            new_request[key] = new_context_roi

        return new_request

    def process(self, batch: gp.Batch, request: gp.BatchRequest):
        """
        Blur batch going downstream for by convolution with kernel defined in init
        and crop to original size. Valid convolution always used.

        :param gp.BatchRequest request: current gp upstream request
        """
        if self.array_key == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_key
            if not isinstance(self.array_key, list):
                keys = [self.array_key]

        blur = np.random.uniform(0, 1) <= self.prob

        if blur:
            for key in keys:
                batch_data = batch[key].data
                channel_dim = -(request[key].roi.dims()) - 1

                if self.blur_channels == None:
                    self.blur_channels = tuple(range(batch_data.shape[channel_dim]))

                output_shape = list(batch_data.shape[:-3]) + list(
                    request[key].roi.get_shape()
                )
                blurred_data = np.empty(output_shape, dtype=batch_data.dtype)

                # TODO: datasets with an additional index beyond channel and batch may
                #      break in loops. Can be safely implemented with dynamic recursion.
                for batch_idx in range(batch_data.shape[0]):
                    for channel_idx in range(batch_data.shape[1]):
                        data = batch_data[batch_idx, channel_idx]
                        if channel_idx in self.blur_channels:
                            data = self._fft_blur(data, self.active_kernel)
                        else:
                            # center crop (I dont trust gp implementation)
                            width = self.active_kernel.shape[0] // 2
                            data = data[:, width:-width, width:-width]

                        blurred_data[batch_idx, channel_idx] = data.astype(
                            batch_data.dtype
                        )

                batch[key] = batch[key].crop(request[key].roi)
                batch[key].data = blurred_data

    def _fft_blur(self, data, kernel):
        """
        Implementation of blurring using FFT. Assumes all data dims are spatial, but
        only blurs along the last two (assumes xy).

        Note: automatically reduces data size as only "valid" convolution is allowed

        :param np.ndarray data: data to blur, > 1 & < 6 dims
        :param np.ndarray kernel: blur kernel to use

        :return np.ndarray data_blur: blurred data
        """
        if len(data.shape) == 5:
            data_blur = signal.fftconvolve(
                data, kernel[np.newaxis, np.newaxis, np.newaxis, :, :], mode="valid"
            )
        elif len(data.shape) == 4:
            data_blur = signal.fftconvolve(
                data, kernel[np.newaxis, np.newaxis, :, :], mode="valid"
            )
        elif len(data.shape) == 3:
            data_blur = signal.fftconvolve(data, kernel[np.newaxis, :, :], mode="valid")
        else:
            data_blur = signal.fftconvolve(data, kernel, mode="valid")
        return data_blur

    def _init_kernels(self):
        """
        Init a kernel for each odd width within width_range.
        """
        # init kernels
        self.kernels = []
        for width in range(self.width_range[0], self.width_range[1] + 1, 2):
            if self.mode == "gaussian":
                t = np.linspace(-10, 10, width)
                bump = np.exp(-self.sigma * t**2)
                bump /= np.trapz(bump)
                self.kernels.append(bump[:, np.newaxis] * bump[np.newaxis, :])
            elif self.mode == "rectangle":
                block = np.ones((width, width))
                self.kernels.append(np.sum(block))
            elif self.mode == "defocus":
                circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
                circle = circle.astype(float)
                circle /= np.sum(circle)
                self.kernels.append(circle)
            else:
                raise AssertionError(f"Mode {self.mode} not an accepted blur mode")


class ChannelSelectionNode(gp.BatchFilter):
    def __init__():
        pass


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
    width_range=(55, 57),
    sigma=0.1,
    prob=1,
    blur_channels=(0,),
)
shear_aug = ShearAugment(
    array=raw,
    angle_range=(-30, 30),
    prob=0.2,
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
        blur_aug,
        cache,  # important to cache upstream of stack
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

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for row in range(3):
        for channel in range(3):
            if channel == 0:
                ax[row][channel].imshow(np.mean(data[row][channel], 0), cmap="gray")
            else:
                ax[row][channel].imshow(data[row][channel][3])
    plt.show()

# %%
