"""
This script align the z-focus of z-stacks to account for any mis-registration in z caused by
 light paths or z-drift during imaging. z-stacks are aligned across different imaging modalities
(label-free & fluorescence), FOVs, times. The focus within each (channel group, p, t) is found
and then aligned. The first channel of each channel group is the reference channel that used to
define the focus of the entire group. Only single-paged tiff input & output are supported for now.
"""

import os
import warnings
import cv2
import numpy as np
import pandas as pd


def brenner_gradient(im):
    assert len(im.shape) == 2, 'Input image must be 2D'
    return np.mean((im[:-2, :] - im[2:, :]) ** 2)


def read_img(img_file):
    """read a single image at (c,t,p,z)"""
    img = cv2.imread(img_file, -1) # flag -1 to preserve the bit dept of the raw image
    if img is None:
        warnings.warn('image "{}" cannot be found. Return None instead.'.format(img_file))
    else:
        img = img.astype(np.float32, copy=False)  # convert to float32 without making a copy to save memory
    return img


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
        im_name += "_z" + str(slice_idx.astype('int64')).zfill(int2str_len)
    if extra_field is not None:
        im_name += "_" + extra_field
    im_name += ext

    return im_name


def main(input_dir,
        output_dir,
        ref_chans,
        chan_groups,
        conditions,
        conditions_new,
        max_focus_idx,
        min_focus_idx):

    assert len(conditions) == len(conditions_new), 'length mismatch for "conditions" and "conditions_new"'
    condi_mapping = dict(zip(conditions, conditions_new))  # {'': ''}
    meta_master = pd.DataFrame()
    for condition in conditions:
        print('processing condition {}...'.format(condition))
        # Load frames metadata and determine indices if exists
        fmeta_path = os.path.join(input_dir, condition, 'frames_meta.csv')
        if os.path.isfile(fmeta_path):
            frames_meta = pd.read_csv(fmeta_path, index_col=0)
        else:
            raise FileNotFoundError('"frames_meta.csv" generated by microDL is required')

        # print(frames_meta['pos_idx'].unique())
        dst_dir = os.path.join(output_dir, condi_mapping[condition])
        os.makedirs(dst_dir, exist_ok=True)

        pos_ids = frames_meta['pos_idx'].unique()
        pos_ids.sort()
        frames_meta['condition'] = condi_mapping[condition]  # empty
        # loop through reference stack at each position

        for pos_idx in pos_ids:
            frames_meta_p = frames_meta[frames_meta['pos_idx'] == pos_idx]
            t_ids = frames_meta_p['time_idx'].unique()
            t_ids.sort()
            for t_idx in t_ids:
                frames_meta_pt = frames_meta_p[frames_meta_p['time_idx'] == t_idx]
                focus_idx = None
                for chans in chan_groups:
                    for chan in chans:
                        print(
                            'Processing position {}, time {}, channel {}...'.format(pos_idx, t_idx, chan))
                        frames_meta_ptc = frames_meta_pt[frames_meta_pt['channel_name'] == chan]
                        z_ids = frames_meta_ptc['slice_idx'].unique()
                        z_ids.sort()
                        if chan in ref_chans:
                            focus_scores = []
                            for z_idx in z_ids:
                                frames_meta_ptcz = frames_meta_ptc[frames_meta_ptc['slice_idx'] == z_idx]
                                im_path = os.path.join(frames_meta_ptcz['dir_name'].values[0],
                                                       frames_meta_ptcz['file_name'].values[0])
                                img = read_img(im_path)
                                focus_score = brenner_gradient(img)
                                focus_scores.append(focus_score)
                            if chan == 'Brightfield':
                                focus_idx = z_ids[np.argmin(focus_scores)]
                            else:
                                focus_idx = z_ids[np.argmax(focus_scores)]
                        else:
                           assert focus_idx is not None, 'reference channel must be the first channel in the channel group'
                        if focus_idx <= max_focus_idx and focus_idx >= min_focus_idx:
                            frames_meta.loc[(frames_meta['pos_idx'] == pos_idx) &
                                            (frames_meta_p['time_idx'] == t_idx) &
                                            (frames_meta_pt['channel_name'] == chan), 'focus_idx'] = focus_idx
                            frames_meta.loc[(frames_meta['pos_idx'] == pos_idx) &
                                            (frames_meta_p['time_idx'] == t_idx) &
                                            (frames_meta_pt['channel_name'] == chan), 'focus_score'] = focus_scores
        frames_meta['dst_dir'] = dst_dir
        meta_master = meta_master.append(frames_meta)  #['channel_idx', 'pos_idx', 'slice_idx', 'time_idx', 'channel_name', 'dir_name', 'file_name', 'condition', 'focus_idx', 'focus_score', 'dst_dir']
        # plot focus scores
        # for chan in ref_chans:
        #     frames_meta_c = frames_meta[frames_meta['channel_name'] == chan]
        #     ax = sns.lineplot(data=frames_meta_c, x='slice_idx', y='focus_score', hue='time_idx')
        #     ax.figure.savefig(os.path.join(input_dir, 'focus_scores_{}_{}.png'.format(condition, chan)))
        #     plt.close()
    focus_offset = meta_master['focus_idx'] - int(meta_master['focus_idx'].median())
    z_ids_new = np.arange(z_ids[0] - focus_offset.min(), z_ids[-1] - focus_offset.max() + 1)
    meta_master['slice_idx_new'] = meta_master['slice_idx'] - focus_offset
    meta_master.loc[~meta_master['slice_idx_new'].isna(), 'slice_idx_new'] = \
        meta_master.loc[~meta_master['slice_idx_new'].isna(), 'slice_idx_new'].astype('int64')
    meta_master = meta_master.loc[meta_master['slice_idx_new'].isin(z_ids_new), :]
    meta_master.reset_index(drop=True, inplace=True)
    for row_idx in list(meta_master.index):
        meta_row = meta_master.loc[row_idx]
        if np.isnan(meta_row['slice_idx_new']):
            continue
        im_src_path = os.path.join(meta_row['dir_name'],
                                   meta_row['file_name'])
        im_name_dst = get_sms_im_name(
            time_idx=meta_row['time_idx'],
            channel_name=meta_row['channel_name'],
            slice_idx=meta_row['slice_idx_new'],
            pos_idx=meta_row['pos_idx'],
            ext='.tif',
        )
        os.link(im_src_path,
                os.path.join(meta_row['dst_dir'], im_name_dst))
    meta_master.to_csv(os.path.join(output_dir, 'frames_meta.csv'), sep=',')


if __name__ == '__main__':
    input_dir = '/hpc/projects/comp_micro/projects/HEK/2022_03_15_orgs_nuc_mem_63x_04NA/all_pos_single_page/all_pos_Phase1e-3_Denconv_Nuc8e-4_Mem8e-4_pad15_bg50'
    output_dir = input_dir + '_registered_refmem_min25_max60'
    max_focus_idx = 60  # maximal focus idx, if the focus of the fov is above it will be neglected
    min_focus_idx = 25  # minimal focus idx, if the focus of a fov is above it will be neglected
    conditions = ['']  # name of the sub-folders for multiple condition (well) dataset. Put '' if no subfolder.
    conditions_new = conditions  # new condition names in the output directory
    pol_chans = ['phase']  # list all polarization channels, reference channel must be listed first
    fluor_chans = ['membrane', 'nucleus']  # list all fluorescence channels, reference channel must be listed first
    ref_chans = ['phase', 'membrane']  # choose a reference channel from the polarization group and from the fluorescence group for alignment
    chan_groups = [pol_chans, fluor_chans]

    main(input_dir,
         output_dir,
         ref_chans,
         chan_groups,
         conditions,
         conditions_new,
         max_focus_idx,
         min_focus_idx)