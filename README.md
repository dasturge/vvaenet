# vvaenet

VNet for medical image segmentation with a VAE twist

this package provides a simple interface for building a parameterized UVAENet 
as in the paper, "3D MRI Brain Tumor Segmentation Using Autoencoder 
Regularization", Myronenko, A.

The intent is to run the model on each task of the Medical Image Segmentation Decathlon (MSD),
it has been built for that express purpose and should be generalizable enough for someone with
a knowledge of python and a bit of study of the eisen-ai software package to repurpose for 
testing other models.

## setup

### acquiring the dataset

one will need to download the entirety of the Medical Image Segmentation Decathlon (MSD) dataset.
This repo expects the Task* folders to be in the working directory, and those to be the only files 
prefixed with "Task".  Optionally, you can omit tasks by simply not including them in the folder.  
They should be untarred and the tarfiles should not be present.

download_datasets.sh will accept a file called "drive_keys.txt", which is simply a file with all of 
the google drive IDs of the MSD dataset. This script doesn't always work as google doesn't like 
command line access to public google drives to prevent data scraping. As such, it's better to just
download them manually. Mainly the utility is there if the code is to be run on a server to avoid 
local download followed by ftp. The datasets amount to some 50GB+ of data.

### pip install

the dependencies should mostly be pythonic aside from cuda setup.

```
pip install -r requirements.txt
```

### Dockerfile

the dockerfile is very basic and untested.  Once I have some free time I'll ensure it can reproduce 
the results.

## run

after your folder has the requisite tasks inside, run:

```
python main.py
```

### usage

```
python main.py --help



```