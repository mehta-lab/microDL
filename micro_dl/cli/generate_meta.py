#!/usr/bin/python

import argparse
import os

import micro_dl.utils.aux_utils as aux_utils


def parse_args():
    """
    Parse command line arguments for directory containing files.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help="Path to folder containing all 2D image frames",
    )
    parser.add_argument(
        '--order',
        type=str,
        default="cztp",
        help="The order in which indices are listed in the image file name",
    )
    parser.add_argument(
        '--name_parser',
        type=str,
        default="get_ids_from_imname",
        help="The function in aux_utils that will parse the file name for indices",
    )
    return parser.parse_args()


def meta_generator(args):
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
    # Import name parser
    parse_func = aux_utils.import_func('utils.aux_utils', args.name_parser)

    # Get all image names
    im_names = aux_utils.get_sorted_names(args.input)
    # Create empty dataframe
    frames_meta = aux_utils.make_dataframe(nbr_rows=len(im_names))
    channel_names = []
    # Fill dataframe with rows from image names
    for i in range(len(im_names)):
        kwargs = {"im_name": im_names[i]}
        if args.name_parser == 'get_ids_from_imname':
            kwargs["order"] = args.order
        elif args.name_parser == 'parse_sms_name':
            kwargs["channel_names"] = channel_names
        frames_meta.loc[i] = parse_func(**kwargs)
    # Write metadata
    meta_filename = os.path.join(args.input, 'frames_meta.csv')
    print(meta_filename, frames_meta.shape)
    frames_meta.to_csv(meta_filename, sep=",")


if __name__ == '__main__':
    args = parse_args()
    meta_generator(args)
