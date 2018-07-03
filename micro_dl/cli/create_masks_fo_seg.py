#!/usr/bin/env python
"""Create masks for segmentation"""
import argparse
import os
import pandas as pd
import pickle

from micro_dl.input.gen_masks_seg import MaskCreator

def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default='/data/anitha/label_free_ff/split_images',
                        help='specify the input dir with full path')
    parser.add_argument('--input_channel_id', type=list,
                        help='specify the input channel ids')
    parser.add_argument(
        '--output_dir', type=str,
        default='/data/anitha/label_free_ff/image_tile_256-256_step_64-64_vf-0.15',
        help='specify the output dir with full path'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    #  Hmm.. how to specify a bunch of params to a group
    #  mutually_exclusive_group is not ideal here
    #  here tile_size & step_size belong to one group vs tile_index_fname in other
    group.add_argument('--tile_size', type=list, default=[256, 256],
                       help='specify tile size along each dimension as a list')
    group.add_argument('--step_size', type=list, default=[256, 256],
                       help='specify step size along each dimension as a list')

    group.add_argument(
        '--tile_index_fname', type=str,
        default='/data/anitha/label_free_ff/split_images/timepoint_0/mask_0-1_vf-0.15.pkl',
        help='path to checkpoint file/directory'
    )

    args = parser.parse_args()
    return args