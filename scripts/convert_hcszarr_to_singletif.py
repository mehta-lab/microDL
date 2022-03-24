"""
Convert a zarr store to single page tif files. Choose which channels, time points, positions, and z-ids to include.
"""

import os
import cv2
from waveorder.io.reader import WaveorderReader
import pprint


def write_img(img, output_dir, img_name):
    """only supports recon_order image name format currently"""
    if not os.path.exists(output_dir):  # create folder for processed images
        os.makedirs(output_dir)
    if len(img.shape) < 3:
        cv2.imwrite(os.path.join(output_dir, img_name), img)
    else:
        cv2.imwrite(os.path.join(output_dir, img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_sms_im_name(time_idx=None,
                    channel_name=None,
                    slice_idx=None,
                    pos_idx=None,
                    extra_field=None,
                    ext='.npy',
                    int2str_len=3):
    """
    Create an image name given parameters and extension
    This function is custom for the computational microscopy (SMS)
    group, who has the following file naming convention:
    File naming convention is assumed to be:
        img_channelname_t***_p***_z***.tif
    This function will alter list and dict in place.

    :param int time_idx: Time index
    :param str channel_name: Channel name
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param str extra_field: Any extra string you want to include in the name
    :param str ext: Extension, e.g. '.png'
    :param int int2str_len: Length of string of the converted integers
    :return st im_name: Image file name
    """
    im_name = "img"
    if channel_name is not None:
        im_name += "_" + str(channel_name)
    if time_idx is not None:
        im_name += "_t" + str(time_idx).zfill(int2str_len)
    if pos_idx is not None:
        im_name += "_p" + str(pos_idx).zfill(int2str_len)
    if slice_idx is not None:
        im_name += "_z" + str(slice_idx).zfill(int2str_len)
    if extra_field is not None:
        im_name += "_" + extra_field
    im_name += ext

    return im_name


def main(input_path,
        output_path,
        conditions,
        channels,
        chan_ids,
        z_ids,
        t_ids,
        pos_ids):
    for condition in conditions:
        print('processing condition {}...'.format(condition))
        # pos_zarrs = get_sub_dirs(os.path.join(input_path, condition + '.zarr'))
        dst_dir = os.path.join(output_path, condition.strip('.zarr'))
        os.makedirs(dst_dir, exist_ok=True)
        t_idx = 0
        reader = WaveorderReader(os.path.join(input_path, condition), data_type = 'zarr')
        pp.pprint(reader.stage_positions)
        pp.pprint(reader.reader.hcs_meta)
        for pos_idx in pos_ids:
            img_tcz = reader.get_zarr(position=pos_idx)  # Returns sliceable array that hasn't been loaded into memory
            for t_idx in t_ids:
                img_cz = img_tcz[t_idx]
                for c_idx, chan in zip(chan_ids, channels):
                    img_z = img_cz[c_idx]
                    for z_idx in z_ids:
                        img = img_z[z_idx]
                        print(
                            'Processing position {}, time {}, channel {}, z {}...'.format(pos_idx, t_idx, chan,
                                                                                          z_idx))
                        im_name_dst = get_sms_im_name(
                            time_idx=t_idx,
                            channel_name=chan,
                            slice_idx=z_idx,
                            pos_idx=pos_idx,
                            ext='.tif',
                        )
                        write_img(img, dst_dir, im_name_dst)


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)

    input_path = '/CompMicro/projects/HEK/2021_08_25_LiveHEK_63x_09NA_StainedOrgs'
    output_path = '/CompMicro/projects/HEK/2021_08_25_LiveHEK_63x_09NA_StainedOrgs_tif'
    conditions = ['Timelapse.zarr']  # name of zarr file
    channels = ['Phase3D', 'DRAQ5']  # choose name of target channels
    chan_ids = [0, 2]  # choose channel ids
    z_ids = list(range(81))  # choose z-ids
    t_ids = list(range(0, 100, 10))  # choose time points
    pos_ids = [3, 5, 9, 11, 17]  # choose positions

    main(input_path,
        output_path,
        conditions,
        channels,
        chan_ids,
        z_ids,
        t_ids,
        pos_ids)
