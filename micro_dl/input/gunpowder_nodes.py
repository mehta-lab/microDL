import gunpowder as gp
import numpy as np
import torch
from scipy import fftpack, signal
import cv2


class LogNode(gp.BatchFilter):
    def __init__(self, prefix, log_image_dir=None):
        """Custom gunpowder node for printing data path

        :param str prefix: prefix to print before message (ex: node number)
        :param str log_image_dir: directory to log data to as it travels downstream
                                    by default, is None, and will not log data
        """
        self.prefix = prefix
        self.log_image_dir = log_image_dir

    def prepare(self, request):
        """Prints message during call to upstream request
        :param gp.BatchRequest request: current gp request
        """
        print(f"{self.prefix}\t Upstream provider: {self.upstream_providers[0]}")

    def process(self, batch, request):
        """Prints message during call to downstream request
        :param gp.BatchRequest batch: batch returned by request from memory
        :param gp.BatchRequest request: current gp request
        """
        if self.log_image_dir:
            pass  # TODO implement this using the datalogging utils.
        print(f"{self.prefix}\tBatch going downstream: {batch}")


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


# %%
