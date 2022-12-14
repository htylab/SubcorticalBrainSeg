import pickle
import os
import collections
import time
import logging
import sys
import nibabel as nib
import numpy as np
import glob as glob
from nilearn.image import reorder_img, new_img_like
from .nilearn_custom_utils.nilearn_utils import crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def normalize_data(data):
    data= data/np.max(data)
    return data


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """
    :param image_files:
    :param image_shape:
    :param crop:
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return:
    """

    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"

        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return image_list




def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    subject_id = in_file.split('/')[-1][:-7]
    image = nib.load(os.path.abspath(in_file))
    image_np = normalize_data(image.get_fdata())
    image = nib.Nifti1Image(image_np,image.affine)
    image = fix_shape(image)

    if crop:
        image = crop_img_to(image, crop, copy=True)

    if image_shape:

        return resize(image, new_shape=image_shape, interpolation=interpolation) #for training

    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    image = new_img_like(image, new_data, affine=new_affine)

    return new_img_like(image, new_data, affine=new_affine)

def get_input_image(input):
    if os.path.isfile(input):
        logging.info(f'Read input: {input}')
        input_image = [input]

    else:
        logging.info(f'Looking for nii.gz in {input}')
        input_image = glob.glob(os.path.join(input,'*.nii.gz'))
        if len(input_image) < 1:
            sys.exit('No files found!')
    return input_image
