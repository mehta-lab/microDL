# Data Organization for virutal staining

Here, we document the conventions for storing data, metadata, configs, and models for next version of microDL.

The main goal with the standardized data organization is to reduce the cognitive burden when iterating on computational experiments. A close second is to make it easy to build upon each other's work without undue duplication of effort or data.


TODO: Following diagram captures the planned flow of data and metadata through the new version of the pipeline. 


The pipeline consists of 5 stages.

* Preprocessing:  Parameters are provided via CLI, and stored along with ome-zarr datasets using (iohub)[https://github.com/czbiohub/iohub]. The metadata is in json format that is practical to edit by hand.
* Training: We use pytorch lightning framework for training, which provides [good defaults](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html) for CLI and training configs, and organized tensorboard [logs](https://lightning.ai/docs/pytorch/stable/extensions/logging.html).
* Inference: The inference module does not depend on lightning, but just on pytorch. Parameters are provided with CLI and stored along with ome-zarr datasets like for preprocessing module. 
* Evaluation: Computation of evaluation metrics does require human proof-reading of ground truth. We currently compute evaluation metrics only with numpy, scipy, scikit-image, etc.  How should we provide parameters? via CLI or with a config?
* Deployment: For run-time deployment, we export the model to .onnx format. 

Data generated by above stages is written to four folders in our data heirarchy.
* Data and metdata produced by preprocessing is saved in `datasets`.
* Input to training (config files) and outputs of training (models) are saved in `models`.
* Data, metadata, and metrics produced by inference and evaluation stages is saved in `evaluation`.
* Deployed models are saved in `deployment`.
## Data heirarchy

```
<virtualstaining>
|
|- datasets (registered, deconvolved, preprocessed)
|   |- yyyy_dd_mm_<dataname0>.zarr
|   |- yyyy_dd_mm_<dataname1>.zarr
|    | ...
|- models
|   |-<experiment0>
|   |   |- lightning_logs
|   |   |- training_config0.yml
|   |   |- training_config1.yml
|   |   |- ...
|   |-<experiment1>
|   |   |- lightning_logs
|   |   |- training_config0.yml
|   |   |- training_config1.yml
|   |   |- ...
|- evaluation
|   |-<experiment0>  (match with paths in models)
|   |   |- predictions.zarr
|   |   |- groundtruth
|   |   |- metrics.csv
|   |   |- ...
|   |-<experiment1>
|   |   |- predictions.zarr
|   |   |- groundtruth
|   |   |- metrics.csv
|   |   |- ...
|- deployment
|   |-<experiment0>  (match with paths in models)
|   |   |- model0.onnx
|   |   |- model0.README
|   |   |...

```