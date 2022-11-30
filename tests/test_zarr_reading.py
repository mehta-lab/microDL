#!/usr/bin/python
'''
Script for testing .zarr reading. Compares with .tiff reader to show that output preprocessed tiles and
metadata are the same from both inputs: 
'''

from copy import deepcopy
import numpy as np
import numpy.testing
import pandas as pd
import nose.tools
import os
import shutil
import unittest
import argparse
import glob
import time

import micro_dl.preprocessing.estimate_flat_field as flat_field
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.image_utils as im_utils
import micro_dl.cli.preprocess_script as preprocess_script


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--zarr_dir',
        type=str,
        help='path to directory of zarr files',
    )
    parser.add_argument(
        '--tiff_dir',
        type=str,
        help='path to directory of tiff files',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='path to directory for writing',
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='path to yaml preprocess config file',
    )
    
    args = parser.parse_args()
    return args


def get_meta_idx(frames_metadata,
                 time_idx,
                 channel_idx,
                 slice_idx,
                 pos_idx,
                 row_start,
                 col_start):
    """
    Get row index in metadata dataframe given variable indices

    :param dataframe frames_metadata: Dataframe with column names given below
    :param int time_idx: Timepoint index
    :param int channel_idx: Channel index
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :return: int pos_idx: Row position matching indices above
    """
    frame_idx = frames_metadata.index[
        (frames_metadata['channel_idx'] == int(channel_idx)) &
        (frames_metadata['time_idx'] == int(time_idx)) &
        (frames_metadata["slice_idx"] == int(slice_idx)) &
        (frames_metadata["pos_idx"] == int(pos_idx)) &
        (frames_metadata["row_start"] == int(row_start)) &
        (frames_metadata["col_start"] == int(col_start))
        ].tolist()
    return frame_idx[0]


def get_channel_correspondence(tiff_config, zarr_config):
    """
    Get channels for each data type since indices differ
    """
    tiff_map = tiff_config['channel_map']
    zarr_map = zarr_config['channel_map']
    # Map tiff channel ids, mask channel (3) is same
    zarr_ids = [3]
    tiff_ids = [3]
    for tiff_name in list(tiff_map):
        zarr_name = [s for s in list(zarr_map) if tiff_name in s.lower()][0]
        print("tiff {} idx: {}, matches zarr {} idx: {}".format(
            tiff_name,
            tiff_map[tiff_name],
            zarr_name,
            zarr_map[zarr_name],
        ))
        zarr_ids.append(int(zarr_map[zarr_name]))
        tiff_ids.append(int(tiff_map[tiff_name]))
    return tiff_ids, zarr_ids


def compare_flatfields(tiff_dir, zarr_dir, tiff_ids, zarr_ids):
    """
    Find corresponding flat field images and compare them
    """
    for tiff_channel_idx in tiff_ids:
        tiff_ff_path = im_utils.get_flat_field_path(
            tiff_dir,
            tiff_channel_idx,
            tiff_ids,
        )
        if tiff_ff_path is not None:
            mapped_idx = tiff_ids.index(tiff_channel_idx)
            zarr_channel_idx = zarr_ids[mapped_idx]
            zarr_ff_path = im_utils.get_flat_field_path(
                zarr_dir,
                zarr_channel_idx,
                tiff_ids,
            )
            assert zarr_ff_path is not None, \
                "No corresponding ff for tiff idx {}".format(tiff_channel_idx)
            # Load and compare
            print("Comparing tiff ff {} with zarr {}".format(
                tiff_ff_path,
                zarr_ff_path,
            ))
            tiff_ff = np.load(tiff_ff_path)
            zarr_ff = np.load(zarr_ff_path)
            numpy.testing.assert_array_equal(tiff_ff, zarr_ff)

    
def compare_tiling(tiff_config, zarr_config, tiff_ids, zarr_ids):
    '''
    Compares .npy files output from preprocessing done through reading zarr and tiff
    files. First load metdata csv file to match up channel indices.
    If any .npy files are substantially different, errors

    :param str tiff_config: Config after preprocessing tiff
    :param str zarr_config: Config after preprocessing zarr
    '''
    tiff_tile_dir = tiff_config['tile']['tile_dir']
    zarr_tile_dir = zarr_config['tile']['tile_dir']
    nose.tools.assert_equal(
        len(os.listdir(tiff_tile_dir)),
        len(os.listdir(zarr_tile_dir)),
    )
    # Read metadata
    tiff_meta = aux_utils.read_meta(tiff_tile_dir)
    tiff_meta.reset_index(drop=True, inplace=True)
    zarr_meta = aux_utils.read_meta(zarr_tile_dir)
    zarr_meta.reset_index(drop=True, inplace=True)
    # Check that shape is the same
    nose.tools.assert_equal(tiff_meta.shape, zarr_meta.shape)

    for idx, tiff_row in tiff_meta.iterrows():
        tiff_channel_idx = tiff_row['channel_idx']
        mapped_idx = tiff_ids.index(tiff_channel_idx)
        zarr_channel_idx = zarr_ids[mapped_idx]
        zarr_idx = get_meta_idx(
            frames_metadata=zarr_meta,
            time_idx=tiff_row['time_idx'],
            channel_idx=zarr_channel_idx,
            slice_idx=tiff_row['slice_idx'],
            pos_idx=tiff_row['pos_idx'],
            row_start=tiff_row['row_start'],
            col_start=tiff_row['col_start'],
        )
        zarr_tile_name = zarr_meta.loc[zarr_idx, 'file_name']
        print("{}. Comparing tiff: {} with zarr: {}".format(
            idx,
            tiff_row['file_name'],
            zarr_tile_name,
        ))
        zarr_tile = np.load(os.path.join(zarr_tile_dir, zarr_tile_name))
        tiff_tile = np.load(os.path.join(tiff_tile_dir, tiff_row['file_name']))
        # Check that all tiles are identical
        numpy.testing.assert_array_equal(zarr_tile, tiff_tile)


def get_base_preprocess_config():
    '''
    Get base 2d preprocessing config file
    
    august 1, :mito, nucleus, phase:, 1_control, 2_roseo.., 3_roseo.., 2D data
        /hpc/projects/CompMicro/projects/Rickettsia/2022_RickettsiaAnalysis_Soorya/3_Cell_Image_Preprocessing/VirtualStainingMicroDL_A549_2022_08_1/SinglePageTiffs_TrainingSet2/
        
    Sep 15 
    '''
    base_preprocess_config = {
        'output_dir': None,
        'verbose': 10,
        'input_dir': None,
        'channel_names': ['Phase3D', 'Nucleus-Hoechst', 'Membrane-CellMask'],
        'slice_ids': [8, 9, 10, 11, 12],
        'pos_ids': [0, 1, 12, 24, 25],
        'num_workers': 4,
        'flat_field':
            {'method': 'estimate',
            'flat_field_channels': ['Nucleus-Hoechst', 'Membrane-CellMask']},
        'normalize':
            {'normalize_im': 'slice',
            'min_fraction': 0.1,
            'normalize_channels': [True, True, True]},
        'uniform_struct': True,
        'masks':
            {'channels': ['Nucleus-Hoechst'],
            'str_elem_radius': 3,
            'mask_type': 'otsu',
            'mask_ext': '.png'},
        'make_weight_map': False,
        'tile':
            {'tile_size': [256, 256],
            'step_size': [128, 128],
            'depths': [1, 1, 5],
            'image_format': 'zyx',
            'min_fraction': 0.1},
        'metadata':
            {'order': 'cztp',
            'name_parser': 'parse_sms_name'},
    }
    return base_preprocess_config


def main(zarr_dir=None, tiff_dir=None, output_dir=None, config_path=None):
    print('Testing zarr reading')

    if config_path is not None:
        preprocess_config = aux_utils.read_config(config_path)
    else:
        preprocess_config = get_base_preprocess_config()
    
    zarr_preprocess_config = deepcopy(preprocess_config)
    zarr_preprocess_config['input_dir'] = zarr_dir
    zarr_preprocess_config['output_dir'] = os.path.join(output_dir, 'temp_zarr_tiles')
    
    tiff_preprocess_config = deepcopy(preprocess_config)
    tiff_preprocess_config['file_format'] = 'tiff'
    tiff_preprocess_config['input_dir'] = tiff_dir
    tiff_preprocess_config['output_dir'] = os.path.join(output_dir, 'temp_tiff_tiles')
    tiff_preprocess_config['channel_names'] = ['phase', 'nucleus', 'membrane']
    tiff_preprocess_config['flat_field']['flat_field_channels'] = \
        ['nucleus', 'membrane']
    tiff_preprocess_config['masks']['channels'] = ['nucleus']

    # generate tiles using tiff and zarr
    if not os.path.exists(zarr_preprocess_config['output_dir']):
        print('\t Running zarr preprocessing...',end='')
        zarr_out_config, runtime = preprocess_script.pre_process(zarr_preprocess_config)
        print('Done. Time: {} s'.format(runtime))
        file_name = os.path.join(zarr_preprocess_config['output_dir'], 'preprocess_config.json')
        aux_utils.write_json(zarr_out_config, file_name)
    if not os.path.exists(tiff_preprocess_config['output_dir']):
        print('\t Running tiff preprocessing ...',end='')
        tiff_out_config, runtime = preprocess_script.pre_process(tiff_preprocess_config)
        print('Done. Time: {} s'.format(runtime))
        file_name = os.path.join(tiff_preprocess_config['output_dir'], 'preprocess_config.json')
        aux_utils.write_json(tiff_out_config, file_name)
    
    # Run tests
    print('Running tests on tiles metadata and files')
    # Get output config files
    file_name = os.path.join(zarr_preprocess_config['output_dir'], 'preprocess_config.json')
    zarr_out_config = aux_utils.read_json(file_name)
    file_name = os.path.join(tiff_preprocess_config['output_dir'], 'preprocess_config.json')
    tiff_out_config = aux_utils.read_json(file_name)

    tiff_ids, zarr_ids = get_channel_correspondence(
        tiff_out_config,
        zarr_out_config,
    )
    # Compare flat field images
    compare_flatfields(
        tiff_out_config['flat_field']['flat_field_dir'],
        zarr_out_config['flat_field']['flat_field_dir'],
        tiff_ids,
        zarr_ids,
    )
    # Compare tiles between data formats
    compare_tiling(
        tiff_out_config,
        zarr_out_config,
        tiff_ids,
        zarr_ids,
    )
    print('Tests completed.')
    # # cleanup directories if pass
    # shutil.rmtree(tiff_prep_config['output_dir'])
    # shutil.rmtree(zarr_prep_config['output_dir'])


if __name__ == '__main__':
    args = parse_args()
    # main('/Volumes/comp_micro/projects/Rickettsia/2022_RickettsiaAnalysis_Soorya/2_Cell_Phase_Reconstruction/Analysis_2022_09_15_A549NuclMemRecon/A5493DPhaseRecon/',
    #      '/Volumes/comp_micro/projects/Rickettsia/2022_RickettsiaAnalysis_Soorya/3_Cell_Image_Preprocessing/VirtualStainingMicroDL_A549NuclMem_2022_09_14_15/Data_UnalignedTiffImages_Sep15/',
    #      '/Volumes/comp_micro/projects/virtualstaining/tf_microDL/config/test_config_preprocess_A549MemNuclStain_Set2.yml')
    main(zarr_dir=args.zarr_dir,
         tiff_dir=args.tiff_dir,
         output_dir=args.output_dir,
         config_path=args.config_path)
