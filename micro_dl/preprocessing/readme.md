# Preprocessing

## Format input data for preprocessing

Before preprocessing make sure the z stacked images are aligned to be centered at the focal plane at all positions. If the focal plane in image stacks imaged
at different positions in a plate are at different z levels, align them using the [z alignment script](https://github.com/mehta-lab/microDL/blob/master/scripts/align_z_focus.py).

## Run preprocessing

The main command for preprocessing is:

```buildoutcfg
python micro_dl/cli/preprocess_script.py --config <config path (.yml)>
```

## Preprocessing config parameters

### Specify input data to be used for training the model

Example preprocessing configuration files for [2D U-Net model](https://github.com/mehta-lab/microDL/blob/microDL-documentation/config_files/Preprocessing-config_2DUnet_regression_phase2nucleus.yml) and [2.5D U-Net model](https://github.com/mehta-lab/microDL/blob/microDL-documentation/config_files/Preprocessing-config_2.5DUnet_regression_phase2membrane.yml) are included in [config_files](https://github.com/mehta-lab/microDL/tree/microDL-documentation/config_files) folder.

The following settings can be adjusted in preprocessing using a config file. 
See example in [config_preprocess](https://github.com/mehta-lab/microDL/blob/microDL-documentation/config_files/config_preprocess.yml), and specifically for 2D virtual staining: [2DUnet_regression_phase2nucleus](https://github.com/mehta-lab/microDL/blob/microDL-documentation/config_files/Preprocessing-config_2DUnet_regression_phase2nucleus.yml).

* input_dir: (str) Directory where image data to be preprocessed is located
* output_dir: (str) folder name where all processed data will be written
* verbose: (int) Logging verbosity levels: NOTSET:0, DEBUG:10, INFO:20, WARNING:30, ERROR:40, CRITICAL:50
* channel_ids: (list of ints) specify channel numbers (default is -1 for all indices), find the numbers in input data metadata.
The id numbers can be found in metadata generated for input data. The ids are allocated based on channel names on image name, accounting
them in the order :  numbers --> uppercase alphabets --> lower case alphabets in alphabetical order.
* slice_ids: (int/list) Value(s) of z-indices to be processed (default is -1 for all indices)
* pos_ids: (int/list) Value(s) of FOVs/positions to be processed (default is -1 for all indices)
* time_ids: (int/list) Value(s) of timepoints to be processed (default is -1 for all indices)
* num_workers: (int) Number of workers for multiprocessing
* resize:
  * scale_factor(float/list): Scale factor for resizing 2D frames, e.g. to match resolution in z.
  * resize_3d (bool): If resizing 3D volumes
  * num_slices_subvolume (int): For 3D resizing: number of slices to be included in each subvolume, default=-1, includes all slices in           slice_ids
* correct_flat_field: (bool) perform flatfield correction for specified channels in 2D.

#### Specify parameters for mask generation

* masks:
  * channels: (list of ints) which channel(s) should be used to generate masks from 
    (if > 1, masks are generated on summed image)
  * str_elem_radius: (int) structuring element radius for morphological operations on masks
  * normalize_im (bool): Whether to normalize image before generating masks
  * mask_dir (str): As an alternative to channels/str_element_radius, you can specify a directory
    containing already generated masks (e.g. manual annotations). Masks must match input images in
    terms of shape and indices.
  * csv_name (str): If specifying mask_dir, the directory must contain a csv file matching mask names
    with image names. If left out, the script will look for first a frames_meta.csv,
    second one csv file containing mask names in one column and matched image names in a
    second column.
  * mask_type (str): Method used for binarization. Options are: 'otsu', 'unimodal', 'dataset otsu', 'borders_weight_loss_map' (for segmentation only).
    E.g. choolse unimodal if intensities are not uniform, otsu if signal is uniform.
  * mask_ext (str): Save format of the processed mask images (for visualization as well!). Options: '.png' or '.npy'

### Tile generation of all input and output images used for training

* tile:
  * tile_size: (list of ints) tile width and height in pixels
  * step_size: (list of ints) step size in pixels for each dimension
  * depths: (list of ints) tile z depth for all the channels specified
  * mask_depth: (int) z depth of mask
  * image_format (str): 'zyx' (default) or 'xyz'. Order of tile dimensions
  * train_fraction (float): If specified in range (0, 1), will randomly select that fraction
    of training data in each epoch. It will update steps_per_epoch in fit_generator accordingly.
  * min_fraction: (float) minimum fraction of image occupied by foreground in masks
  * hist_clip_limits: (list of ints) lower and upper intensity percentiles for histogram clipping

The tiling class will take the 2D image files, crop them into tiles given tile and step size,
and store them in .npy format. For 2.5D, the images will be stacked in z, given depth, prior to tiling.
If min_fraction is specified, only tiles with a minimum amount of foreground, determined from masks, 
will be retained.

### Specifications of metadata to be generated

* metadata (for tiff files. Soon most metadata will be found in zarr stores.)
  * order (str): If 'cztp', the images produced by processing are named by the format 'channel_zslice_time_position', for example, 'c001_z046_t000_p023'
  * name_parser (str): 'parse_sms_name' corresponds to 'sms' image naming format 'img_channelname_t***_p***_z***_customfield'.'

All data will be stored in the specified output dir, with a 'preprocessing_info.json' file.
The preprocessing_info.json contains all of the parameters associated with preprocessing, in addition to
metadata about the preprocessing runtime and config directory.

During preprocessing, a csv file named frames_meta.csv will be generated in the tiled data directory. 
The csv contains the following fields for each tile:

* 'time_idx': the timepoint it came from
* 'channel_idx': its channel
* 'slice_idx': the z index in case of 3D data
* 'pos_idx': the field of view index
* 'file_name': file name
* 'row_start': starting row for tile (add tile_size for endpoint)
* 'col_start': start column (add tile_size for endpoint)
* 'dir_name': directory path
