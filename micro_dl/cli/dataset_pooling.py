"""Pool multiple datasets into a single dataset for training"""
import argparse
import os
import yaml
import pandas as pd
import numpy as np
from shutil import copy2
import micro_dl.utils.aux_utils as aux_utils
from micro_dl.utils.image_utils import read_image

def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        help='path to inference yaml configuration file',
    )

    args = parser.parse_args()
    return args

def meta_generator(input_dir, order='cztp', name_parser='parse_sms_name'):
    """
    Generate metadata from file names for preprocessing.
    Will write found data in frames_metadata.csv in input directory.
    Assumed default file naming convention is:
    dir_name
    |
    |- im_c***_z***_t***_p***.png
    |- im_c***_z***_t***_p***.png

    c is channel
    z is slice in stack (z)
    t is time
    p is position (FOV)

    Other naming convention is:
    img_channelname_t***_p***_z***.tif for parse_sms_name

    :param list args:    parsed args containing
        str input_dir:   path to input directory containing images
        str name_parser: Function in aux_utils for parsing indices from file name
    """
    parse_func = aux_utils.import_object('utils.aux_utils', name_parser, 'function')
    im_names = aux_utils.get_sorted_names(input_dir)
    frames_meta = aux_utils.make_dataframe(nbr_rows=len(im_names))
    channel_names = []
    # Fill dataframe with rows from image names
    for i in range(len(im_names)):
        kwargs = {"im_name": im_names[i]}
        if name_parser == 'parse_idx_from_name':
            kwargs["order"] = order
        elif name_parser == 'parse_sms_name':
            kwargs["channel_names"] = channel_names
        meta_row = parse_func(**kwargs)

        im_path = os.path.join(input_dir, im_names[i])
        im = read_image(im_path)
        meta_row['dir_name'] = input_dir
        meta_row['mean'] = np.nanmean(im)
        meta_row['std'] = np.nanstd(im)
        frames_meta.loc[i] = meta_row

    # Write metadata
    meta_filename = os.path.join(input_dir, 'frames_meta.csv')
    frames_meta.to_csv(meta_filename, sep=",")
    return frames_meta

def pool_dataset(config):
    """
    :param dict args: dict with input options
    :return:
    """

    config_fname = config
    with open(config_fname, 'r') as f:
        pool_config = yaml.safe_load(f)
    # Import name parser

    dst_dir = pool_config['destination']
    frames_meta_dst_path = os.path.join(dst_dir, 'frames_meta.csv')
    pos_idx_cur = 0
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(frames_meta_dst_path):
        frames_meta_dst = pd.read_csv(frames_meta_dst_path, index_col=0)
        pos_idx_cur = frames_meta_dst['pos_idx'].max() + 1
    else:
        frames_meta_dst = aux_utils.make_dataframe(nbr_rows=None)

    for src_key in pool_config:
        if 'source' in src_key:
            src_dir = pool_config[src_key]['dir']
            src_pos_ids = pool_config[src_key]['pos_ids']
            frames_meta_src = meta_generator(src_dir,
                                             name_parser=pool_config['name_parser'])
            if src_pos_ids == 'all':
                src_pos_ids = frames_meta_src['pos_idx'].unique()
            src_pos_ids.sort()
            frames_meta_src_new = frames_meta_src.copy()
            # select positions to pool and update their indices
            frames_meta_src_new = frames_meta_src_new[frames_meta_src['pos_idx'].isin(src_pos_ids)]
            pos_idx_map = dict(zip(src_pos_ids, range(pos_idx_cur, pos_idx_cur + len(src_pos_ids))))
            frames_meta_src_new['pos_idx'] = frames_meta_src_new['pos_idx'].map(pos_idx_map)
            # frames_meta_src_new['dir_name'] = dst_dir

            # update file names and copy the files
            for row_idx in list(frames_meta_src_new.index):
                meta_row = frames_meta_src_new.loc[row_idx]
                im_name_dst = aux_utils.get_sms_im_name(time_idx=meta_row['time_idx'],
                                                        channel_name=meta_row['channel_name'],
                                                        slice_idx=meta_row['slice_idx'],
                                                        pos_idx=meta_row['pos_idx'],
                                                        ext='.tif',
                                                        )
                frames_meta_src_new.loc[row_idx, 'file_name'] = im_name_dst
                im_name_src = frames_meta_src.loc[row_idx, 'file_name']
                copy2(os.path.join(src_dir, im_name_src),
                      os.path.join(dst_dir, im_name_dst))

            frames_meta_dst = frames_meta_dst.append(
                frames_meta_src_new,
                ignore_index=True,
            )
            pos_idx_cur = pos_idx_map[src_pos_ids[-1]] + 1
    frames_meta_dst.to_csv(frames_meta_dst_path, sep=",")

if __name__ == '__main__':
    args = parse_args()
    pool_dataset(args.config)
