"""Stich model predictions either along Z or as tiles"""
import numpy as np


class ImageStitcher:
    """Stitch prediction images for 3D

    USE PREDICT ON LARGER IMAGE FOR 2D AND 2.5D
    """

    def __init__(self, tile_option,
                 overlap_dict,
                 image_format='zxy',
                 data_format=None):
        """Init

        :param str tile_option: 'tile_z' or 'tile_xyz'
        :param dict overlap_dict: with keys overlap_shape, overlap_operation
        and z_dim. overlap_shape is an int for tile_z and list of len
         3 for tile_xyz. z_dim is an int corresponding to z dimension
        """

        assert tile_option in ['tile_z', 'tile_xyz'], \
            'tile_option not in [tile_z, tile_xyz]'

        allowed_overlap_opn = ['mean', 'any']
        assert ('overlap_operation' in  overlap_dict and
                overlap_dict['overlap_operation'] in allowed_overlap_opn), \
            'overlap_operation not provided or not in [mean, any]'
        assert image_format in ['zxy', 'xyz'], 'image_format not in [zxy, xyz]'

        self.tile_option = tile_option
        self.overlap_dict = overlap_dict
        self.data_format = data_format

        img_dim = [2, 3, 4] if self.data_format == 'channels_first' \
            else [1, 2, 3]
        self.img_dim = img_dim
        if data_format == 'channels_first':
            x_dim = 3 if image_format == 'zxy' else 2
        elif data_format == 'channels_last':
            x_dim = 2 if image_format == 'zxy' else 1
        y_dim = x_dim + 1

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = overlap_dict['z_dim']
        self.image_format = image_format

    def _place_block_z(self,
                       pred_block,
                       pred_image,
                       start_idx,
                       end_idx):
        """Place the current block prediction in the larger vol

        :param np.array pred_block: current prediction block
        :param np.array pred_image: full 3D prediction image with zeros
        :param int start_idx: start slice of pred_block
        :param int end_idx: end slice of pred block
        :return np.array pred_image: with pred_block placed accordingly
        """

        num_overlap = self.overlap_dict['overlap_shape']
        overlap_operation = self.overlap_dict['overlap_operation']
        z_dim = self.overlap_dict['z_dim']

        # smoothly weight the two images in overlapping slices
        forward_wts = np.linspace(0, 1.0, num_overlap + 2)[1:-1]
        reverse_wts = forward_wts[::-1]
        # initialize all indices to :
        idx_in_img = []
        idx_in_block = []
        for dim_idx in range(len(pred_image).shape):
            idx_in_img.append(np.s_[:])
            idx_in_block.append(np.s_[:])
        idx_in_img[z_dim] = np.sl_[start_idx + num_overlap : end_idx]
        idx_in_block[z_dim] = np.sl_[num_overlap:]
        pred_image[idx_in_img] = pred_block[idx_in_block]
        if start_idx > 0:
            for sl_idx in range(num_overlap):
                idx_in_img[z_dim] = start_idx + sl_idx
                idx_in_block[z_dim] = sl_idx
                if overlap_operation == 'mean':
                    pred_image[idx_in_img] = \
                        (forward_wts[sl_idx] * pred_image[idx_in_img] +
                         reverse_wts[sl_idx] * pred_block[idx_in_block])
                elif overlap_operation == 'any':
                    pred_block[idx_in_img] = np.any(pred_image[idx_in_img],
                                                    pred_block[idx_in_block])
        else:
            idx_in_img[z_dim] = np.sl_[start_idx : start_idx + num_overlap]
            idx_in_block[z_dim] = np.sl_[0: num_overlap]
            pred_image[idx_in_img] = pred_block[idx_in_block]
        return pred_image

    def _stitch_along_z(self,
                        tile_imgs_list,
                        block_indices_list):
        """Stitch images along Z with or w/o overlap"""

        stitched_img = np.zeros(self.shape_3d)

        assert 'z_dim' in self.overlap_dict, \
            'z_dim was not provided in overlap_dict'
        if 'overlap_shape' in self.overlap_dict:
            assert isinstance(self.overlap_dict['overlap_shape'], int), \
                'tile_z only supports an overlap of int slices along z'

        for idx, sub_block in enumerate(tile_imgs_list):
            try:
                cur_sl_idx = block_indices_list[idx]
                stitched_img = self._place_block_z(pred_block=sub_block,
                                                   pred_image=stitched_img,
                                                   start_idx=cur_sl_idx[0],
                                                   end_idx=cur_sl_idx[1])
            except Exception as e:
                e.args += 'error in _stitch_along_z'
                raise
        return stitched_img

    def _place_block_xyz(self,
                         pred_block,
                         pred_image,
                         crop_index):
        """Place the current block prediction in the larger vol

        :param np.array pred_block:
        :param np.array pred_image:
        :param list crop_index:
        """

        overlap_shape = self.overlap_dict['overlap_shape']
        overlap_operation = self.overlap_dict['overlap_operation']

        # smoothly weight the two images in overlapping slices
        forward_wts = np.linspace(0, 1.0, overlap_shape + 2)[1:-1]
        reverse_wts = forward_wts[::-1]
        # initialize all indices to :
        idx_in_img = []
        idx_in_block = []
        for dim_idx in range(len(pred_image).shape):
            idx_in_img.append(np.s_[:])
            idx_in_block.append(np.s_[:])

        # assign non-overlapping regions
        for idx, dim_idx in enumerate(self.img_dim):
            idx_in_block[dim_idx] = np.sl_[crop_index[idx * 2]:
                                           crop_index[idx * 2 + 1]]
            idx_in_img[dim_idx] = np.sl_[overlap_shape[idx]:]
        pred_image[idx_in_img] = pred_block[idx_in_block]
        # overlap along left and top borders, top now

        overlap_dim = [self.x_dim, self.y_dim, self.z_dim]
        for dim_idx, cur_dim in enumerate(overlap_dim):
            # 0 - zdim (front), 1 - xdim (top), 2 - ydim (left) if zyx
            idx_in_block[cur_dim] = np.sl_[0: overlap_shape[dim_idx]]
            if overlap_shape[dim_idx] > 0:
                for idx in range(overlap_shape[dim_idx]):
                    idx_in_img[cur_dim] = (
                        np.sl_[crop_index[2 * dim_idx]:
                               crop_index[2 * dim_idx] + overlap_shape[dim_idx]]
                    )
                    if overlap_operation == 'mean':
                        pred_image[idx_in_img] = \
                            (forward_wts[idx] * pred_image[idx_in_img] +
                             reverse_wts[idx] * pred_block[idx_in_block])
                    elif overlap_operation == 'any':
                        pred_block[idx_in_img] = np.any(
                            pred_image[idx_in_img], pred_block[idx_in_block]
                        )
            else:
                idx_in_img[cur_dim] = (
                    np.sl_[crop_index[2 * dim_idx]:
                           crop_index[2 * dim_idx] + overlap_shape[dim_idx]]
                )
                pred_image[idx_in_img] = pred_block[idx_in_block]
        return pred_image

    def _stitch_along_xyz(self,
                          tile_imgs_list,
                          block_indices_list):
        """Stitch images along XYZ with overlap"""

        stitched_img = np.zeros(self.shape_3d)
        assert self.data_format is not None, \
            'data format needed for stitching images along xyz'
        for idx, cur_tile in tile_imgs_list:
            try:
                cur_crop_idx = block_indices_list[idx]
                stitched_img = self._place_block_xyz(pred_block=cur_tile,
                                                     pred_image=stitched_img,
                                                     crop_index=cur_crop_idx)
            except Exception as e:
                e.args += 'error in _stitch_along_z'
                raise
        return stitched_img

    def stitch_predictions(self, shape_3d,
                           tile_imgs_list,
                           block_indices_list):
        """Stitch the predicted tiles /blocks for a 3d image

        :param list shape_3d: shape of  3d image
        :param list tile_imgs_list: list of prediction images
        :param list block_indices_list: list of tuple/lists with indices for
         each prediction. Individual list of: len=2 when tile_z (start_slice,
         end_slice), len=6 for tile_xyz with start and end indices for each
         dimension
        """

        assert len(tile_imgs_list) == len(block_indices_list), \
            'missing tile/indices for sub tile/block: {}, {}'.format(
                len(tile_imgs_list), len(block_indices_list)
            )
        assert len(shape_3d) == 3, \
            'only stitching 3D volume is currently supported'
        self.shape_3d = shape_3d

        if self.tile_option == 'tile_z':
            stitched_img = self._stitch_along_z(tile_imgs_list,
                                                block_indices_list)
        elif self.tile_option == 'tile_xyz':
            pass
        return stitched_img
