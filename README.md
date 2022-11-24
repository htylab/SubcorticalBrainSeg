#### This repo is for methods describing in the paper.
Weng JS, Huang TY, “Deriving a robust deep-learning model for subcortical brain segmentation by using a large-scale database: Preprocessing, reproducibility, and accuracy of volume estimation” (2022), NMR in Biomedicine

#### For updated models, please visit: https://github.com/htylab/tigerbx

# Automated subcortical brain segmentation pipeline

## Background
This package provides trained 3D U-Net model for subcortical brain segmentation


## Tutorial using SubBrainSegment

### Install package

    pip install https://github.com/htylab/tigerseg/archive/main.zip 

## Usage

### As a command line tool:

    tigerseg INPUT OUTPUT

If INPUT points to a file, the file will be processed. If INPUT points to a directory, the directory will be searched for the specific format(nii.gz).
OUTPUT is the output directory.

For additional options type:

    tigerseg -h



### As a python module:

```
import tigerseg.segment

input_dir = /your/input/directory
output_dir = /your/output/directory

tigerseg.segment.apply(input=input_dir,output=output_dir)
```