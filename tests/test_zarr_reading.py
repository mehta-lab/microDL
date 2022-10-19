#!/usr/bin/python
'''
Script for testing .zarr reading. Compares with .tiff reader to show that output preprocessed tiles and
metadata are the same from both inputs: 
'''

from copy import deepcopy
import numpy as np
import pandas as pd
import os
import shutil
import unittest
import argparse
import glob
import time

import micro_dl.preprocessing.estimate_flat_field as flat_field
import micro_dl.utils.aux_utils as aux_utils
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


class TestZarrReading(unittest.TestCase):

    def setUp(self):
        """
        Write .yml files for preprocessing with .zarr and .tif from the same directory
        Specify a directory of .zarr files.
        """
        self.zarr_tiles = None
        self.tiff_tiles = None
    
    def compare_tiling(self, tiff_tile_dir, zarr_tile_dir):
        '''
        Compares .npy files output from preprocessing done through reading zarr and tiff 
        files. If any .npy files are substantially different, errors
        
        :param str tiff_tile_dir: directory containing tiff-preprocessed tiles
        :param str zarr_tile_dir: directory containing zarr-preprocessed tiles
        '''
        tiff_tile_names = self._get_files_of_type(tiff_tile_dir, ftype='.npy')
        zarr_tile_names = self._get_files_of_type(zarr_tile_dir, ftype='.npy')
        
        assert len(tiff_tile_names) == len(zarr_tile_names), 'Different numbers of tiles'
        
        for i in range(len(tiff_tile_names)):
            tiff_tile = np.load(os.path.join(tiff_tile_dir, tiff_tile_names[i]))
            zarr_tile = np.load(os.path.join(zarr_tile_dir, zarr_tile_names[i]))
            print(f'Comparing tile {i}/{len(zarr_tile_names)}...', end='\r')
            time.sleep(0.00001)
            
            er_msg = f'Tiles {zarr_tile_names[i]} (zarr) and {tiff_tile_names[i]} (tiff) unequal'
            np.testing.assert_equal(tiff_tile, zarr_tile, err_msg = er_msg)
        print('\n', end='\r')
        print('\t --> compare_tiling passed')

    def compare_metadata(self, tiff_tile_dir, zarr_tile_dir):
        '''
        Compares metadata files output from preprocessing done through reading zarr and tiff 
        data storage. If metadata is different, errors.
        
        :param str tiff_tile_dir: directory containing tiff-preprocessed tiles
        :param str zarr_tile_dir: directory containing zarr-preprocessed tiles
        '''
        tiff_meta_files = self._get_files_of_type(tiff_tile_dir, ftype='.csv')
        zarr_meta_files = self._get_files_of_type(zarr_tile_dir, ftype='.csv')
        for i in range(len(tiff_meta_files)):
            tiff_meta = pd.read_csv(tiff_meta_files[i])
            zarr_meta = pd.read_csv(zarr_meta_files[i])
            print(f'Comparing metadata {i+1}/{len(tiff_meta_files)}...', end='\r')
            # Remove directories since they're not the same
            tiff_meta.drop(['dir_name'], axis=1)
            zarr_meta.drop(['dir_name'], axis=1)
            try:
                tiff_meta.compare(zarr_meta)
            except:
                er_msg = f'Metadata {zarr_meta_files[i]} (zarr) and {tiff_meta_files[i]} (tiff) unequal'
                raise AssertionError(er_msg)
        print('\n', end='\r')
        print('\t --> compare_metadata passed')
    
    def _get_files_of_type(self, root_dir, ftype='.csv'):
        '''
        Recursively search for and return all filenames of type 'ftype' in
        directory or subdirectories of 'root_dir'
        :return list all_csv_files: List of all files
        '''
        PATH = root_dir
        EXT = '*' + ftype
        all_csv_files = [y for x in os.walk(PATH) for y in glob.glob(os.path.join(x[0], EXT))]
        all_csv_files.sort(key=str)
        return all_csv_files


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
        zarr_prep_config, runtime = preprocess_script.pre_process(zarr_preprocess_config)
        print('Done. Time: {} s'.format(runtime))
    if not os.path.exists(tiff_preprocess_config['output_dir']):
        print('\t Running tiff preprocessing ...',end='')
        tiff_prep_config, runtime = preprocess_script.pre_process(tiff_preprocess_config)
        print('Done. Time: {} s'.format(runtime))
    
    # initiate tester and run tests
    tester = TestZarrReading()
    
    try:
        print('Running tests...')
        # compare methods
        tester.compare_tiling(tiff_prep_config['output_dir'], zarr_prep_config['output_dir'])
        tester.compare_metadata(tiff_prep_config['output_dir'], zarr_prep_config['output_dir'])
        
        # # cleanup directories if pass
        # shutil.rmtree(tiff_prep_config['output_dir'])
        # shutil.rmtree(zarr_prep_config['output_dir'])
        
    except Exception as e:
        print(e.args)
    

if __name__ == '__main__':
    args = parse_args()
    # main('/Volumes/comp_micro/projects/Rickettsia/2022_RickettsiaAnalysis_Soorya/2_Cell_Phase_Reconstruction/Analysis_2022_09_15_A549NuclMemRecon/A5493DPhaseRecon/',
    #      '/Volumes/comp_micro/projects/Rickettsia/2022_RickettsiaAnalysis_Soorya/3_Cell_Image_Preprocessing/VirtualStainingMicroDL_A549NuclMem_2022_09_14_15/Data_UnalignedTiffImages_Sep15/',
    #      '/Volumes/comp_micro/projects/virtualstaining/tf_microDL/config/test_config_preprocess_A549MemNuclStain_Set2.yml')
    main(zarr_dir=args.zarr_dir,
         tiff_dir=args.tiff_dir,
         output_dir=args.output_dir,
         config_path=args.config_path)
