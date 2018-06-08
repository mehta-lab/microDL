#!/usr/bin/python

import argparse

import micro_dl.input.preprocess_images as preprocess_images
import micro_dl.utils.aux_utils as aux_utils


def parse_args():
    """
    Parse command line arguments for data preprocessing
    prior to training U-Net model

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help="Path to folder containing all time/channel subfolders")
    parser.add_argument('-o', '--output', type=str,
                        help=("Path to base directory where preprocessed data ",
                              "and info csv file will be written"))
    parser.add_argument('-d', '--data_split', nargs=3, type=float, default=[.6, .2, .2],
                        help=("Specify fractions of data you want to use for ",
                              "training, validation and testing"))
    parser.add_argument('--tile_size', nargs='*', type=int, default=[256, 256],
                        help="Split images into tiles of size [height, width (depth)]")
    parser.add_argument('--step_size', nargs='*', type=int, default=[256, 256],
                        help="Split images into patches of size [height, width]")
    parser.add_argument('-v', '--verbose', type=int, default=10)
    return parser.parse_args()


def preprocess(args):
    """
    Split, crop volumes and flatfield correct images in input and target
    directories. Writes output as npy files for faster reading while training.

    :param list args:    parsed args containing
        str input_dir:   path to input main directory containing subfolders
                         for timepoints containing subfolders for channels
        str output_dir:  base path where processed data will be written
        list data_split: fractions of train, validation and test (must sum to 1)
        list tile_size:  shape of image tiles
        list step_size:  shape of step size when making image tiles
        int verbose:     verbosity of preprocess
    """
    # Instantiate preprocessor
    preprocessor = preprocess_images.ImagePreprocessor(
        args.input,
        args.output,
        args.verbose)
    # Input folder should contain timepoint folders. Each timepoint should
    # contain channel folders with channel number (int)
    # in the folder name. Each channel folder should contain unique matching
    # indices. Check and collect indices, then save images as npy
    channel_nbrs, im_indices = preprocessor.channel_validator()
    preprocessor.save_images_as_npy(
        channel_nbrs,
        im_indices,
        mask_channels=None,
        num_timepoints=1)

    # Normalize and tile images
    preprocessor.crop_images(
        args.tile_size,
        args.step_size,
        channel_ids=-1,
        mask_channel_ids=None,
        min_fraction=None)

    # crop_params = {}
    # if meta_preprocess['split_volumes']:
    #     if 'mask_channels' in meta_preprocess:
    #         preprocessor.save_images(meta_preprocess['input_fname'],
    #                                  meta_preprocess['mask_channels'],
    #                                  meta_preprocess['focal_plane_idx'])
    #         crop_params['mask_channel_ids'] = meta_preprocess['mask_channels']
    #         crop_params['focal_plane_idx'] = meta_preprocess['focal_plane_idx']
    #         crop_params['min_fraction'] = meta_preprocess['min_fraction']
    #     else:
    #         preprocessor.save_images(meta_preprocess['input_fname'])
    #
    # if 'isotropic' in meta_preprocess['crop_volumes']:
    #     crop_params['isotropic'] = meta_preprocess['crop_volumes']['isotropic']
    #
    # if meta_preprocess['crop_volumes']:
    #     preprocessor.crop_images(
    #         tile_size=meta_preprocess['crop_volumes']['tile_size'],
    #         step_size=meta_preprocess['crop_volumes']['step_size'],
    #         channel_ids=meta_preprocess['crop_volumes']['channels'],
    #         **crop_params
    #     )


if __name__ == '__main__':
    args = parse_args()
    preprocess(args)
