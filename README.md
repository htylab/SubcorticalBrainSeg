#### This repo is for the methods described in the paper.
Weng JS, Huang TY, “Deriving a robust deep-learning model for subcortical brain segmentation by using a large-scale database: Preprocessing, reproducibility, and accuracy of volume estimation” (2022), NMR in Biomedicine


* This repo is only for GPU-based segmentation.
* For updated models or CPU-based segmentation, please visit: https://github.com/htylab/tigerbx

# Automated subcortical brain segmentation pipeline

## Background
This package provides trained 3D U-Net model for subcortical brain segmentation


## Tutorial using SubBrainSegment

### Install package

    pip install https://github.com/htylab/SubcorticalBrainSeg/archive/main.zip 

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


#### We gratefully acknowledge the authors of the U-Net implementation:
https://github.com/ellisdg/3DUnetCNN
