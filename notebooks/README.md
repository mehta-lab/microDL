# Exercise 4: Image translation

## Setup

If the `(base)` prefix is not present in front of the shell prompt, you need to initialize conda and restart the terminal:
```
conda init bash
```

Open the terminal and copy the data to the home directory:

```
mkdir 04_image_translation
cp -r /mnt/efs/woods_hole/04_image_translation_data ~
```


Download `microDL` repository:

```
git clone https://github.com/czbiohub/microDL.git
```

Go the repository directory, switch the branch: 

```
cd microDL
git checkout dl_mbl_2021
```

Create a `conda` environment for this exercise from the yaml file and activate it:

```
conda env create --file=conda_environment.yml
conda activate micro_dl
```

Add microDL to python path:

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Next, launch a jupyter notebook server, if you're running the notebook from a server: 

```
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Then you can access your notebooks in your browser at:

```
http://<your server name>:8888
```

Enter the token jupyter generated when you launched the notebook, open the notebook under /microDL/notebook/, 
and continue with the instructions in the notebook.