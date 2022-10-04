# Inference

## Run inference

The main command for inference is:

```buildoutcfg
python micro_dl/cli/inference_script.py --config <config path (.yml)> --gpu <gpu id (default 0)> --gpu_mem_frac <0-1 (default 1>
```

where the parameters are defined as follows:

* **config** (yaml file): Configuration file, see below.
* **gpu** (int): ID number of if you'd like to specify which GPU you'd like to run on. If you don't
specify a GPU then the GPU with the largest amount of available memory will be selected for you.
* **gpu_mem_fraction** (float): You can specify what fraction of total GPU memory you'd like to utilize.
If there's not enough memory available on the GPU, and AssertionError will be raised.
If memory fraction is unspecified, all memory currently available on the GPU will automatically
be allocated for you.

## Inference config settings

Example inference configuration files for [2D U-Net model](https://github.com/mehta-lab/microDL/blob/microDL-documentation/config_files/Inference-config_2DUnet_regression_phase2nucleus.yml) and [2.5D U-Net model](https://github.com/mehta-lab/microDL/blob/microDL-documentation/config_files/Inference-config_2.5DUnet_regression_phase2membrane.yml) are available in [config_files](https://github.com/mehta-lab/microDL/tree/microDL-documentation/config_files) folder.

The following settings can be adjusted in inference using a config file. See example in [config_inference](https://github.com/mehta-lab/microDL/blob/microDL-documentation/config_files/config_inference.yml).yml and [Inference-config_2DUnet_regression_phase2nucleus](https://github.com/mehta-lab/microDL/blob/microDL-documentation/config_files/Inference-config_2DUnet_regression_phase2nucleus.yml).yml. The config file consists of three main parts, *images*, *metrics* and *inference_3d*, where the latter only needs to be specified if running prediction on 3D images.

The images part is focused around running predictions, whereas metrics parameters are specified if you also would like to generate evaluation metrics. If you've already run inference and generated your prediction images you can also evaluate metrics independently using the metrics_script.

* preprocess_dir (str): path to preprocess dir with the preprocessing_info.json file
* model_dir (str): Path to model directory
* model_fname (str/None): File name of weights in model dir (.hdf5).
If left out, latest weights file will be selected.
* image_dir (str): Directory containing full size input images (not tiles)
* data_split (str): Which data (train/test/val) to run inference on.
 (default = test)
* save_figs (str): 'True' or 'False', if figures generated should be saved.

* dataset:
  * input_channels (list/None): label-free channel used for prediction by model, get numbers from metadata
  * target_channels (list/None): target image channel (fluorescence image) to compare how well the prediction worked
  * slice_ids (list/None): image slices
  * pos_ids (list/None): may not effect the positions where prediction is performed if data split is defined
* images:
  * image_format (str): 'zyx' or 'xyz' for depth dimension first or last
  * flat_field_dir (str/None): Directory containing flatfield images
  * im_ext (str): For writing generated prediction images, e.g. '.png' or '.npy' or '.tiff'. For 3D images the only option is '.npy'.
  * crop_shape (list): Center crop the image to a specified shape before inference. If None, leave images as is.
  * save_to_image_dir (True/False): 'False' saves output in model directory, 'True' in input image directory
  * image_ext (str): saved output image extenstion, example is '.tif', output images are stored as single page tiff files.
  * name_format (str): 'sms' corresponds to image naming format 'img_channelname_t***_p***_z***_customfield', default naming convention is 'im_c***_z***_t***_p***'
  * pred_chan_name (str): prefix of saved output image name
* masks
  * mask_dir (str): Mask directory containing a frames_meta.csv with
    mask channels (which will be target channels in the inference config)
    z, t, p indices matching the ones in image_dir. Mask dirs are often either
    generated or have had a frames_meta added to them during preprocessing.
  * mask_type (str): 'target' for segmentation, 'metrics' for weighted metrics
  * mask_channel (int): mask channel as in training
* inference_3d: params if doing 3D inference:
  * num_slices (int): in case of 3D, the full volume will not fit in GPU
    memory, specify the number of slices to use and this will depend on
    the network depth, for ex 8 for a network of depth 4.
  * inf_shape (list): Inference on a center sub volume.
  * tile_shape (list): Shape of tile for tiling along xyz.
  * num_overlap (int/list): int for tile_z, list for tile_xyz
  * overlap_operation (str): e.g. 'mean'
* metrics: metrics to be computed to define prediction quality, the values will be printed on output figure panel and saved as text files
  * metrics_list (list): List of metrics to estimate. Currently available metrics:
    * 'ssim' - structural similarity index
    * 'corr' - correlation
    * 'r2' - coefficient of determination
    * 'mse' - mean squared error
    * 'mae' - mean absolute error
    * 'dice' - dice similarity coefficient (for binary data, no masks)
    * 'acc' - accuracy of binary target & prediction (for binary data, no masks)
    * metrics_orientations (list): Assuming images are of shape xyz you can evaluate metrics
    along any number of given planes 'xy', 'xz', 'yz' as well as generating global 3D metrics
    using 'xyz'.

## Prediction metrics

If you have already generated your predictions and would like to generate evaluation metrics
you can use the metrics script:

```buildoutcfg
python micro_dl/cli/metrics_script.py --model_dir <path> --image_dir <path> --ext <file type> --metrics <types> --orientations <slices along xyz>
```

With the following parameters:

* **model_dir** (str): Model directory. Assumed to contain config, split_samples.json and a subdirectory
named 'predictions' that already contains predictions.
* **test_data**: An optional (default) flag that specifies that only test indices in split_samples.json
will be evaluated. The other option is to use all the data with the flag **all_data**.
* **image_dir** (str): Directory containing target images of same size as predictions.
* **metrics** (list): See inference -> metrics_list for options.
* **orientations**: Any subset of {xy, xz, yz, xyz}, see inference -> metrics_orientations.
