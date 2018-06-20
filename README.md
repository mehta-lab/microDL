# MicroDL

This is a pipeline for training U-Net models. It consists of three modudules:

* Preprocessing: normalization, flatfield correction, masking, tiling
* Training: 2D-3D model creation, training with certain losses, metrics, learning rates
* Inference: Perform inference on tiles that can be stitched to full images

## Getting Started

To run preprocessing on a Lif file, see config_preprocess.yml file
and then run

```
python micro_dl/input/preprocess_script.py --config micro_dl/config_preprocess.yml
```

To run preprocessing on images, they need to be in the following folder structure

```
input_dir
    |
    |-timepoint0
        |
        |-channel0
        |-channel1
        |-...
    |
    |-timepoint1
        |
        |-channel0
        |-channel1
        |-...
    |
    |-...
```

To train the model, you can modify the followin config file and run:

```
python micro_dl/train/train_script.py --config micro_dl/config.yml
```