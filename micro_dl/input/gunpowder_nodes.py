import cv2
import gunpowder as gp
import numpy as np
from scipy import fftpack, signal
import skimage
import torch
import time

from micro_dl.input.dataset import apply_affine_transform
from micro_dl.utils.normalize import unzscore


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
        shear_middle_slice_channels=None,
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
        :param tuple(int)/None shear_middle_slice_channels: shear only the middle z-slice of these
                                                    channel indices. Used when target plucked
                                                    from stack. By default None.
        """
        assert (
            abs(angle_range[0]) <= 30 and abs(angle_range[1]) <= 30
        ), "bounds of shearing angle"
        f" range must be withing [-30, 30] but are {angle_range}"

        self.array_keys = array
        self.angle_range = angle_range
        self.prob = prob
        self.shear_middle_slice_channels = shear_middle_slice_channels

    def prepare(self, request: gp.BatchRequest):
        """
        Prepare request going upstream for data retrieval: increases request size
        in rows dimension to accommodate for information lost in corners during shearing.

        :param gp.BatchRequest request: current gp downstream request

        :return gp.BatchRequest request: modified or unmodified request depending on
                                        random roll > threshold of self.prob
        """
        if self.array_keys == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_keys
            if not isinstance(self.array_keys, list):
                keys = [self.array_keys]

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
        Shear batch going downstream using shear matrix to apply affine
        transformation. Interpolate between pixels

        Only applies shear if randomely determined to in node preparation.
        Applies transformation to all arrays in self.array_keys

        :param gp.BatchRequest request: current gp downstream request
        :param gp.Batch batch: current batch traveling downstream
        """
        if self.array_keys == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_keys
            if not isinstance(self.array_keys, list):
                keys = [self.array_keys]

        if self.apply_shear:
            for key in keys:
                batch_data = batch[key].data
                roi = request[key].roi

                if self.shear_middle_slice_channels == None:
                    self.shear_middle_slice_channels = ()

                output_shape = list(batch_data.shape[:-3]) + list(
                    request[key].roi.get_shape()
                )
                sheared_data = np.empty(output_shape, dtype=batch_data.dtype)

                # TODO: datasets with an additional index beyond channel and batch may
                #      break in loops. Can be safely implemented with dynamic recursion.
                for batch_idx in range(batch_data.shape[0]):
                    for channel_idx in range(batch_data.shape[1]):
                        middle_only = channel_idx in self.shear_middle_slice_channels
                        data = batch_data[batch_idx, channel_idx]

                        if roi.dims() == 2:
                            data = np.expand_dims(data, 0)
                            data = apply_affine_transform(data, shear=self.angle)[0]
                        else:
                            if middle_only:
                                middle_idx = data.shape[0] // 2
                                data[middle_idx] = apply_affine_transform(
                                    np.expand_dims(data[middle_idx], 0),
                                    shear=self.angle,
                                )[0]
                            else:
                                data = apply_affine_transform(data, shear=self.angle)

                        if self.angle > 0:
                            data = data[
                                ..., :, : data.shape[-1] - self.extra_pixels * 2
                            ]
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

        :param gp.ArrayKey or list arrays: key(s) to data to perform blurring on. This is provided for
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

        self.array_keys = array
        self.mode = mode
        self.width_range = width_range
        self.prob = prob
        self.sigma = sigma
        self.blur_channels = blur_channels

        self._init_kernels()

        if self.blur_channels == None:
            Warning("You are blurring all channels. This is likely a mistake...")

    def prepare(self, request: gp.BatchRequest):
        """
        Prepare request going upstream for data retrieval by increasing request size
        to accommodate for kernel width. This is to ensure valid convolution of region.

        :param gp.BatchRequest request: current gp upstream request
        """
        if self.array_keys == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_keys
            if not isinstance(self.array_keys, list):
                keys = [self.array_keys]

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
        if self.array_keys == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_keys
            if not isinstance(self.array_keys, list):
                keys = [self.array_keys]

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


class IntensityAugment(gp.BatchFilter):
    def __init__(
        self,
        array=None,
        jitter_channels=None,
        scale_range=(0.7, 1.3),
        shift_range=(-0.3, 0.3),
        norm_before_shift=False,
        jitter_demeaned=True,
        prob=1,
    ):
        """
        Custom gunpowder augmentation node for applying intensity jitter by volume or
        slice.

        Intensity jitter scales and shifts the intensity values of an entire FoV according
        to:
                a = a.mean() + ((a-a.mean()) * scale) + shift

        The scale/shift values are selected from the given ranges, but the trans-
        formation is standard across each image.

        Assumes channel dimension is last dimension before spatial dimensions.

        :param gp.ArrayKey array: key to array to perform jittering on. If no key provided,
                                applies to all key:data pairs in request.
        :param tuple(int) jitter_channels: jitter intensity of only these channels in
                                            channel dimension
        :param tuple(int, int) scale_range: range of random scale
        :param tuple(int, int) shift_range: range of random shift
        :param bool norm_before_shift: normalize the data to [0,1] before applying shift. Will
                                        renormalize after, but treats shift as a proportion
                                        of max contrast than numeric value
        :param bool jitter_demeaned: If true, only applies jitter shift to the demeaned
                                    components to augment contrast over absolute intensity
        :param float prob: probability of applying jitter
        """
        assert (
            scale_range[0] >= 0 and scale_range[1] >= 0
        ), "Scale bounds must be positive"
        assert scale_range[0] <= scale_range[1], "Scale bounds must be non-decreasing"
        assert shift_range[0] <= shift_range[1], "Shift bounds must be non-decreasing"

        self.array_key = array
        self.jitter_channels = jitter_channels
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.norm_before_shift = norm_before_shift
        self.jitter_demeaned = jitter_demeaned
        self.prob = prob

        if jitter_channels == None:
            Warning(
                "You are jittering the intensity of all channels. This is likely a mistake..."
            )

    def prepare(self, request: gp.BatchRequest):
        """
        Prepare request going upstream. sets random seet
        """
        np.random.seed(request.random_seed)
        self.active_scale = np.random.uniform(*self.scale_range)
        self.active_shift = np.random.uniform(*self.shift_range)

    def process(self, batch: gp.Batch, request: gp.BatchRequest):
        """
        Jitter batch going downstream according to parameters.

        :param gp.BatchRequest request: current gp downstream request
        :param gp.Batch batch: current batch heading downstream
        """
        if self.array_key == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_key
            if not isinstance(self.array_key, list):
                keys = [self.array_key]

        jitter = np.random.uniform(0, 1) <= self.prob

        if jitter:
            for key in keys:
                batch_data = batch[key].data
                channel_dim = -(request[key].roi.dims()) - 1

                if self.jitter_channels == None:
                    self.jitter_channels = tuple(range(batch_data.shape[channel_dim]))

                output_shape = list(batch_data.shape[:-3]) + list(
                    request[key].roi.get_shape()
                )
                jittered_data = np.empty(output_shape, dtype=batch_data.dtype)

                # TODO: datasets with an additional index beyond channel and batch may
                #      break in loops. Can be safely implemented with dynamic recursion.
                for batch_idx in range(batch_data.shape[0]):
                    for channel_idx in range(batch_data.shape[1]):
                        data = batch_data[batch_idx, channel_idx]
                        if channel_idx in self.jitter_channels:
                            data = self._jitter(
                                data, self.active_scale, self.active_shift
                            )

                        jittered_data[batch_idx, channel_idx] = data.astype(
                            batch_data.dtype
                        )

                batch[key] = batch[key].crop(request[key].roi)
                batch[key].data = jittered_data

    def _jitter(self, a, scale, shift):
        """
        Perform actual jitter computation

        :param np.ndarray a: array to apply intensity jitter to
        :param float scale: demeaned scaling factor
        :param float shift: shifting factor

        :return np.ndarray jittered_array: see name

        """
        if self.norm_before_shift:
            max_contrast = np.max(a - np.min(a))
            shift = max_contrast * shift

        if self.jitter_demeaned:
            return a.mean() + unzscore(a - a.mean(), shift, scale)
        else:
            return unzscore(a, shift, scale)


class NoiseAugment(gp.BatchFilter):
    def __init__(
        self,
        array=None,
        mode="gaussian",
        noise_channels=None,
        seed=None,
        clip=True,
        prob=1,
        **kwargs,
    ):
        """
        Custom gunpowder augmentation node for applying random noise.
        For kwargs see skimage.util.randomnoise:
            https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise

        Assumes channel dim is the last non-spatial dimension.

        :param gp.ArrayKey array: key to array to perform jittering on. If no key provided,
                                applies to all key:data pairs in request.
        :param str mode: type of noise to apply. see skimage.util.randomnoise
        :param tuple(int) noise_channels: noise only these channels in channel dimension,
                                            by default applies to all channels
        :param int seed: Optionally set a random seed, see scikit-image documentation.
        :param bool clip: Whether to preserve the image range after adding noise or
                            not, (note: noise will be scaled to image intensity values)
        :param float prob: probability of applying noise
        """

        self.array_key = array
        self.mode = mode
        self.noise_channels = noise_channels
        self.seed = seed
        self.clip = clip
        self.prob = prob
        self.kwargs = kwargs

        if noise_channels == None:
            Warning(
                "You are applying noise to all channels. This is likely a mistake..."
            )

    def prepare(self, request: gp.BatchRequest):
        """
        Prepare request going upstream. sets random seet
        """
        if not self.seed:
            self.seed = np.random.seed(request.random_seed)

    def process(self, batch: gp.Batch, request: gp.BatchRequest):
        """
        Noise batch going downstream according to parameters.

        :param gp.BatchRequest request: current gp downstream request
        :param gp.Batch batch: current batch heading downstream
        """
        if self.array_key == None:
            keys = [pair[0] for pair in request.items()]
        else:
            keys = self.array_key
            if not isinstance(self.array_key, list):
                keys = [self.array_key]

        noise = np.random.uniform(0, 1) <= self.prob

        if noise:
            for key in keys:
                batch_data = batch[key].data
                channel_dim = -(request[key].roi.dims()) - 1

                if self.noise_channels == None:
                    self.noise_channels = tuple(range(batch_data.shape[channel_dim]))

                output_shape = list(batch_data.shape[:-3]) + list(
                    request[key].roi.get_shape()
                )
                noisy_data = np.empty(output_shape, dtype=batch_data.dtype)

                # TODO: datasets with an additional index beyond channel and batch may
                #      break in loops. Can be safely implemented with dynamic recursion.
                for batch_idx in range(batch_data.shape[0]):
                    for channel_idx in range(batch_data.shape[1]):
                        data = batch_data[batch_idx, channel_idx]
                        if channel_idx in self.noise_channels:
                            data = self._apply_noise(data)

                        noisy_data[batch_idx, channel_idx] = data.astype(
                            batch_data.dtype
                        )

                batch[key] = batch[key].crop(request[key].roi)
                batch[key].data = noisy_data

    def _apply_noise(self, a):
        """
        Apply noise using skimage.util.randomnoise

        :param np.ndarray a: array to apply noise to
        :return np.ndarray jittered_array: see name

        """
        # normalize
        norm_val = np.max(np.abs(a))
        a = a / norm_val

        # noise
        a = skimage.util.random_noise(
            a,
            mode=self.mode,
            seed=self.seed,
            clip=self.clip,
            **self.kwargs,
        ).astype(a.dtype)

        # denormalize
        a = (a * norm_val).astype(a.dtype)

        return a
