import os
import pandas as pd
import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.mp_utils as mp_utils
import itertools

def frames_meta_generator(
        input_dir,
        order='cztp',
        name_parser='parse_sms_name',
        num_workers=4
        ):
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
    mp_fn_args = []
    mp_block_args = []
    block_size = 256
    # Fill dataframe with rows from image names
    for i in range(len(im_names)):
        kwargs = {"im_name": im_names[i]}
        if name_parser == 'parse_idx_from_name':
            kwargs["order"] = order
        elif name_parser == 'parse_sms_name':
            kwargs["channel_names"] = channel_names
        meta_row = parse_func(**kwargs)
        meta_row['dir_name'] = input_dir
        frames_meta.loc[i] = meta_row
        im_path = os.path.join(input_dir, im_names[i])
        mp_fn_args.append(im_path)
        mp_block_args.append((im_path, block_size, meta_row))

    im_stats_list = mp_utils.mp_get_im_stats(mp_fn_args, num_workers)
    im_stats_df = pd.DataFrame.from_dict(im_stats_list)
    frames_meta[['mean', 'std']] = im_stats_df[['mean', 'std']]

    im_blocks_list = mp_utils.mp_sample_im_blocks(mp_block_args, num_workers)
    im_blocks_list = list(itertools.chain.from_iterable(im_blocks_list))
    blocks_meta = pd.DataFrame.from_dict(im_blocks_list)

    # Write metadata
    frames_meta_filename = os.path.join(input_dir, 'frames_meta.csv')
    frames_meta.to_csv(frames_meta_filename, sep=",")

    blocks_meta_filename = os.path.join(input_dir, 'blocks_meta.csv')
    blocks_meta.to_csv(blocks_meta_filename, sep=",")
    return frames_meta


