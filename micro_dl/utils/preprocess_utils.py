import glob
import os

import micro_dl.utils.aux_utils as aux_utils


def validate_mask_meta(pp_config):
    """
    If user provides existing masks, the mask directory should also contain
    a csv file (not named frames_meta.csv which is reserved for output) with
    two column names: mask_name and file_name. Each row should describe the
    mask name and the corresponding file name. Each file_name should exist in
    input_dir and belong to the same channel.
    This function checks that all file names exist in input_dir and writes a
    frames_meta csv containing mask names the indices corresponding to the
    matched file_name. It also assigns a mask channel number for future
    preprocessing steps like tiling.

    :param dict pp_config: Preprocessing config
    :return int mask_channel: New channel index for masks for writing tiles
    :raises AssertionError: If 'masks' in pp_config contains both channels
        and mask_dir (the former is for generating masks from a channel)
    :raises IOError: If no csv file is present in mask_dir
    :raises AssertionError: If more than one csv file exists in mask_dir
        and no csv_name is provided to resolve ambiguity
    :raises AssertionError: If provided csv_name if 'frames_meta.csv' - that
        name is reserved for preprocessing output
    :raises AssertionError: If csv doesn't consist of two columns named
        'mask_name' and 'file_name'
    :raises IndexError: If unable to match file_name in mask_dir csv with
        file_name in input_dir for any given mask row
    :raises AssertionError: If mask file correspond to more than one input
        channel
    """
    assert 'channels' not in pp_config['masks'], \
        "Don't specify channels to mask if using pre-generated masks"
    mask_dir = pp_config['masks']['mask_dir']
    # Look for a csv. If more than one, get name from config
    csv_name = glob.glob(os.path.join(mask_dir, '*.csv'))
    if len(csv_name) == 0:
        raise IOError("No csv file present in mask dir")
    elif len(csv_name) > 1:
        assert 'csv_name' in pp_config['masks'], \
            "Please add csv_name to config->masks to resolve ambiguity"
        csv_name = pp_config['masks']['csv_name']
    else:
        # Use the one existing csv name
        csv_name = csv_name[0]
        assert csv_name != 'frames_meta.csv',\
            "frames_meta.csv is reserved for output. Please rename mask csv."
    mask_meta = aux_utils.read_meta(input_dir=mask_dir, meta_fname=csv_name)

    assert len(set(mask_meta).difference({'file_name', 'mask_name'})) == 0,\
        "mask csv should have columns mask_name and file_name " +\
        "(corresponding to the file_name in input_dir)"
    # Check that file_name for each mask_name matches files in input_dir
    input_dir = pp_config['input_dir']
    input_meta = aux_utils.read_meta(input_dir)
    file_names = input_meta['file_name']
    # Create dataframe that will store all indices for masks
    out_meta = aux_utils.make_dataframe(nbr_rows=mask_meta.shape[0])
    for i, row in mask_meta.iterrows():
        try:
            file_loc = file_names[file_names == row.file_name].index[0]
        except IndexError as e:
            msg = "Can't find image file name match for {}, error {}".format(
                row.file_name, e)
            raise msg
        # Fill dataframe with row indices from matched image in input dir
        out_meta.iloc[i] = input_meta.iloc[file_loc]
        # Write back the mask name
        out_meta.iloc[i]['file_name'] = row.mask_name

    assert len(out_meta.channel_idx.unique()) == 1,\
        "Masks should match one input channel only"
    # Write mask metadata with indices that match input images
    meta_filename = os.path.join(mask_dir, 'frames_meta.csv')
    out_meta.to_csv(meta_filename, sep=",")
    # Masks need to have their own channel index for tiling
    # Hopefully this will be big enough default value
    mask_channel = 999
    if 'channel_ids' in pp_config:
        mask_channel = max(pp_config['channel_ids']) + 1

    return mask_channel



