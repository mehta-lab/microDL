"""Estimate flat field images"""

import numpy as np
import os
import zarr

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as im_utils
import micro_dl.utils.io_utils as io_utils


class FlatFieldEstimator2D:
    """Estimates flat field image"""

    def __init__(
        self,
        zarr_dir,
        channel_ids,
        slice_ids,
        block_size=32,
        flat_field_array_name="flatfield",
    ):
        """
        Flatfield images are estimated once per channel for 2D data.

        Flatfields are estimated by averaging over all dataset positions to capture
        static perturbations. Flatfields are stored in an additional array at the
        image-level of the input HCS compatible zarr store.

        Images can be corrected by dividing by their channel's flatfield on the fly.

        :param str zarr_dir: HCS Compatible zarr directory
        :param int/list channel_ids: channel ids for flat field_correction
        :param int/list slice_ids: Z slice indices for flatfield correction
        :param int block_size: Size of blocks image will be divided into
        """
        self.zarr_dir = zarr_dir
        self.flat_field_array_name = flat_field_array_name
        self.slice_ids = slice_ids
        self.channels_ids = channel_ids

        # get meta
        metadata_ids = aux_utils.validate_metadata_indices(
            zarr_dir=self.zarr_dir,
            channel_ids=channel_ids,
            slice_ids=slice_ids,
        )
        self.channels_ids = metadata_ids["channel_ids"]
        self.slice_ids = metadata_ids["slice_ids"]

        if block_size is None:
            block_size = 32
        self.block_size = block_size

        # initate modifier
        self.modifier = io_utils.HCSZarrModifier(zarr_file=zarr_dir)

    def get_flat_field_dir(self):
        """
        Return flatfield directory
        :return str flat_field_dir: Flatfield directory
        """
        return self.zarr_dir

    def get_flat_field_name(self):
        """
        Return name of flat field array in each well
        :return str flat_field_name: see name
        """
        return self.flat_field_array_name

    def get_hyperparameters(self):
        """
        Group hyperparameters from this flatfield estimator and return as a dictionary
        """
        metadata = {
            "array_name": self.flat_field_array_name,
            "channel_ids": self.channels_ids,
            "slice_ids": self.slice_ids,
            "block_size": self.block_size,
        }
        return metadata

    def estimate_flat_field(self):
        """
        Estimates flat field correction image and stores in zarr store at the image level
        as a new array.

        Records hyperparameters used in estimation in .zattrs metadata.
        """
        # flat_field constant over time, so use first time idx. And use only first
        # slice if multiple are present
        time_idx = 0
        slice_idx = 0
        all_channels_array = []

        for channel_idx in self.channels_ids:
            summed_image = None

            # Average over all positions
            for position in list(self.modifier.position_map):
                for slice_idx in self.slice_ids:
                    position_zarr = self.modifier.get_zarr(position)
                    im = position_zarr[time_idx, channel_idx, slice_idx]

                    if len(im.shape) == 3:
                        im = np.mean(im, axis=2)
                    if summed_image is None:
                        summed_image = im.astype("float64")
                    else:
                        summed_image += im

            mean_image = summed_image / len(list(self.modifier.position_map))
            # TODO (Jenny): it currently samples median values from a mean
            # images, not very statistically meaningful but easier than
            # computing median of image stack
            flatfield = self.get_flatfield(mean_image)
            all_channels_array.append(flatfield)

        all_channels_array = np.stack(all_channels_array, 0)
        all_channels_array = np.expand_dims(all_channels_array, (0, 2))

        # record flat_field inside zarr store.
        for position in self.modifier.position_map:
            self.modifier.init_untracked_array(
                data_array=all_channels_array,
                position=position,
                name=self.flat_field_array_name,
            )
            self.modifier.write_meta_field(
                position=position,
                metadata=self.get_hyperparameters(),
                field_name="flatfield",
            )

    def sample_block_medians(self, im):
        """Subdivide a 2D image in smaller blocks of size block_size and
        compute the median intensity value for each block. Any incomplete
        blocks (remainders of modulo operation) will be ignored.

        :param np.array im:         2D image
        :return np.array(float) sample_coords: Image coordinates for block
                                               centers
        :return np.array(float) sample_values: Median intensity values for
                                               blocks
        """

        im_shape = im.shape
        assert self.block_size < im_shape[0], "Block size larger than image height"
        assert self.block_size < im_shape[1], "Block size larger than image width"

        nbr_blocks_x = im_shape[0] // self.block_size
        nbr_blocks_y = im_shape[1] // self.block_size
        sample_coords = np.zeros((nbr_blocks_x * nbr_blocks_y, 2), dtype=np.float64)
        sample_values = np.zeros((nbr_blocks_x * nbr_blocks_y,), dtype=np.float64)
        for x in range(nbr_blocks_x):
            for y in range(nbr_blocks_y):
                idx = y * nbr_blocks_x + x
                sample_coords[idx, :] = [
                    x * self.block_size + (self.block_size - 1) / 2,
                    y * self.block_size + (self.block_size - 1) / 2,
                ]
                sample_values[idx] = np.median(
                    im[
                        x * self.block_size : (x + 1) * self.block_size,
                        y * self.block_size : (y + 1) * self.block_size,
                    ]
                )
        return sample_coords, sample_values

    def get_flatfield(self, im, order=2, normalize=True):
        """
        Combine sampling and polynomial surface fit for flatfield estimation.
        To flatfield correct an image, divide it by flatfield.

        :param np.array im:        2D image
        :param int order:          Order of polynomial (default 2)
        :param bool normalize:     Normalize surface by dividing by its mean
                                   for flatfield correction (default True)

        :return np.array flatfield:    Flatfield image
        """

        coords, values = self.sample_block_medians(im=im)
        flatfield = im_utils.fit_polynomial_surface_2D(
            sample_coords=coords,
            sample_values=values,
            im_shape=im.shape,
            order=order,
            normalize=normalize,
        )
        # Flatfields can't contain zeros or negative values
        if flatfield.min() <= 0:
            raise ValueError(
                "The generated flatfield was not strictly positive {}.".format(
                    flatfield.min()
                ),
            )
        return flatfield
