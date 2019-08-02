import json
import nose.tools
from testfixtures import TempDirectory

import micro_dl.utils.aux_utils as aux_utils


# Create test metadata table
meta_df = aux_utils.make_dataframe()
channel_idx = 5
time_idx = 6
for s in range(3):
    for p in range(4):
        im_name = aux_utils.get_im_name(
            channel_idx=channel_idx,
            slice_idx=s,
            time_idx=time_idx,
            pos_idx=p,
            ext='.png',
        )
        meta_df = meta_df.append(
            aux_utils.parse_idx_from_name(im_name),
            ignore_index=True,
        )


def test_get_row_idx():
    row_idx = aux_utils.get_row_idx(meta_df, time_idx, channel_idx)
    nose.tools.assert_equal(row_idx.all(), True)


def test_get_row_idx_slice():
    row_idx = aux_utils.get_row_idx(meta_df, time_idx, channel_idx, slice_idx=1)
    for i, val in row_idx.items():
        if meta_df.iloc[i].slice_idx == 1:
            nose.tools.assert_true(val)
        else:
            nose.tools.assert_false(val)


def test_get_row_idx_slice_pos():
    row_idx = aux_utils.get_row_idx(
        meta_df,
        time_idx,
        channel_idx,
        slice_idx=0,
        pos_idx=3,
    )
    for i, val in row_idx.items():
        if meta_df.iloc[i].slice_idx == 0 and meta_df.iloc[i].pos_idx == 3:
            nose.tools.assert_true(val)
        else:
            nose.tools.assert_false(val)


def test_get_sorted_names():
    with TempDirectory() as tempdir:
        # The function doesn't care about file format, names just have to start with im_
        empty_json = {}
        tempdir.write('im_10.json', json.dumps(empty_json).encode())
        tempdir.write('im_5.json', json.dumps(empty_json).encode())
        file_names = aux_utils.get_sorted_names(tempdir.path)
        nose.tools.assert_equal(len(file_names), 2)
        nose.tools.assert_equal(file_names[0], 'im_5.json')
        nose.tools.assert_equal(file_names[1], 'im_10.json')


def test_parse_idx_from_name():
    im_name = 'im_c005_z010_t50000_p020.png'
    meta = aux_utils.parse_idx_from_name(im_name, order="cztp")
    nose.tools.assert_equal(meta["channel_idx"], 5)
    nose.tools.assert_equal(meta["slice_idx"], 10)
    nose.tools.assert_equal(meta["time_idx"], 50000)
    nose.tools.assert_equal(meta["pos_idx"], 20)


def test_parse_idx_different_order():
    im_name = 'im_t50000_z010_p020_c005.png'
    meta = aux_utils.parse_idx_from_name(im_name, order="tzpc")
    nose.tools.assert_equal(meta["channel_idx"], 5)
    nose.tools.assert_equal(meta["slice_idx"], 10)
    nose.tools.assert_equal(meta["time_idx"], 50000)
    nose.tools.assert_equal(meta["pos_idx"], 20)

@nose.tools.raises(AssertionError)
def test_parse_idx_from_name_no_channel():
    file_name = 'img_phase_t500_p400_z300.tif'
    aux_utils.parse_idx_from_name(file_name)


def test_parse_sms_name():
    file_name = 'img_phase_t500_p400_z300.tif'
    channel_names = ['brightfield']
    meta_row = aux_utils.parse_sms_name(file_name, channel_names=channel_names)
    nose.tools.assert_equal(channel_names, ['brightfield', 'phase'])
    nose.tools.assert_equal(meta_row['channel_name'], 'phase')
    nose.tools.assert_equal(meta_row['channel_idx'], 1)
    nose.tools.assert_equal(meta_row['time_idx'], 500)
    nose.tools.assert_equal(meta_row['pos_idx'], 400)
    nose.tools.assert_equal(meta_row['slice_idx'], 300)


def test_parse_sms_name_long_channel():
    file_name = 'img_long_c_name_t001_z002_p003.tif'
    channel_names = []
    meta_row = aux_utils.parse_sms_name(file_name, channel_names=channel_names)
    nose.tools.assert_equal(channel_names, ['long_c_name'])
    nose.tools.assert_equal(meta_row['channel_name'], 'long_c_name')
    nose.tools.assert_equal(meta_row['channel_idx'], 0)
    nose.tools.assert_equal(meta_row['time_idx'], 1)
    nose.tools.assert_equal(meta_row['pos_idx'], 3)
    nose.tools.assert_equal(meta_row['slice_idx'], 2)
