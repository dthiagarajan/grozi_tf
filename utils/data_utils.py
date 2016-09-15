# -*- coding: utf-8 -*-
"""
Preprocessing provides some useful functions to preprocess data before
training, such as pictures dataset building, sequence padding, etc...
Note: Those preprocessing functions are only meant to be directly applied to
data, they are not meant to be use with Tensors or Layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
from PIL import Image


_EPSILON = 1e-8


# =======================
# TARGETS (LABELS) UTILS
# =======================


def to_categorical(y, nb_classes):
    """ to_categorical.
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


# ===================
#    IMAGES UTILS
# ===================

def build_hdf5_image_dataset(target_path, image_shape, output_path='dataset.h5',
                             mode='file', categorical_labels=True,
                             normalize=True, grayscale=False,
                             files_extension=None, chunks=True):
    """ Build HDF5 Image Dataset.
    Build an HDF5 dataset by providing either a root folder or a plain text
    file with images path and class id.
    'folder' mode: Root folder should be arranged as follow:
    ```
    ROOT_FOLDER -> SUBFOLDER_0 (CLASS 0) -> CLASS0_IMG1.jpg
                                         -> CLASS0_IMG2.jpg
                                         -> ...
                -> SUBFOLDER_1 (CLASS 1) -> CLASS1_IMG1.jpg
                                         -> ...
                -> ...
    ```
    Note that if sub-folders are not integers from 0 to n_classes, an id will
    be assigned to each sub-folder following alphabetical order.
    'file' mode: Plain text file should be formatted as follow:
    ```
    /path/to/img1 class_id
    /path/to/img2 class_id
    /path/to/img3 class_id
    ```
    Examples:
        ```
        # Load path/class_id image file:
        dataset_file = 'my_dataset.txt'
        # Build a HDF5 dataset (only required once)
        from tflearn.data_utils import build_hdf5_image_dataset
        build_hdf5_image_dataset(dataset_file, image_shape=(128, 128),
                                 mode='file', output_path='dataset.h5',
                                 categorical_labels=True, normalize_img=True)
        # Load HDF5 dataset
        import h5py
        h5f = h5py.File('dataset.h5', 'w')
        X = h5f['X']
        Y = h5f['Y']
        # Build neural network and train
        network = ...
        model = DNN(network, ...)
        model.fit(X, Y)
        ```
    Arguments:
        target_path: `str`. Path of root folder or images plain text file.
        image_shape: `tuple (height, width)`. The images shape. Images that
            doesn't match that shape will be resized.
        output_path: `str`. The output path for the hdf5 dataset. Default:
            'dataset.h5'
        mode: `str` in ['file', 'folder']. The data source mode. 'folder'
            accepts a root folder with each of his sub-folder representing a
            class containing the images to classify.
            'file' accepts a single plain text file that contains every
            image path with their class id.
            Default: 'folder'.
        categorical_labels: `bool`. If True, labels are converted to binary
            vectors.
        normalize: `bool`. If True, normalize all pictures by dividing
            every image array by 255.
        grayscale: `bool`. If true, images are converted to grayscale.
        files_extension: `list of str`. A list of allowed image file
            extension, for example ['.jpg', '.jpeg', '.png']. If None,
            all files are allowed.
        chunks: `bool` or `list of int`. Whether to chunks the dataset or not.
            Additionaly, a specific shape for each chunk can be provided.
    """
    import h5py

    assert image_shape, "Image shape must be defined."
    assert image_shape[0] and image_shape[1], \
        "Image shape error. It must be a tuple of int: ('width', 'height')."
    assert mode in ['folder', 'file'], "`mode` arg must be 'folder' or 'file'"

    if mode == 'folder':
        images, labels = directory_to_samples(target_path,
                                              flags=files_extension)
    else:
        with open(target_path, 'r') as f:
            images, labels = [], []
            for l in f.readlines():
                l = l.strip('\n').split(' ')
                images.append(l[0])
                labels.append(l[1])

    n_classes = np.max(labels) + 1

    d_imgshape = (len(images), image_shape[0], image_shape[1], 3) \
        if not grayscale else (len(images), image_shape[0], image_shape[1])
    d_labelshape = (len(images), n_classes) \
        if categorical_labels else (len(images), )

    dataset = h5py.File(output_path, 'w')
    dataset.create_dataset('X', d_imgshape, chunks=chunks)
    dataset.create_dataset('Y', d_labelshape, chunks=chunks)

    for i in range(len(images)):
        img = load_image(images[i])
        # width, height = img.size
        # if width != image_shape[0] or height != image_shape[1]:
        #     img = resize_image(img, image_shape[0], image_shape[1])
        # if grayscale:
        #     img = convert_color(img, 'L')
        img = pil_to_nparray(img)
        if normalize:
            img /= 255.
        dataset['X'][i] = img
        if categorical_labels:
            dataset['Y'][i] = to_categorical([labels[i]], n_classes)[0]
        else:
            dataset['Y'][i] = labels[i]


def load_image(in_image):
    """ Load an image, returns PIL.Image. """
    img = Image.open(in_image)
    return img


def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array.

    Args:
        pil_image (PIL.Image.Image):
    """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


def image_dirs_to_samples(directory, filetypes=None):
    print("Starting to parse images...")
    if filetypes and filetypes not in [list, tuple]:
        filetypes = list(filetypes)
    samples, targets = directory_to_samples(directory, flags=filetypes)
    for i, s in enumerate(samples):
        samples[i] = load_image(s)
        samples[i] = pil_to_nparray(samples[i])
        samples[i] /= 255.
    print("Parsing Done!")
    return samples, targets


def build_image_dataset_from_dir(directory,
                                 dataset_file="my_tflearn_dataset.pkl",
                                 filetypes=None, shuffle_data=False,
                                 categorical_Y=False):
    try:
        X, Y = pickle.load(open(dataset_file, 'rb'))
    except Exception:
        X, Y = image_dirs_to_samples(directory, filetypes)
        if categorical_Y:
            Y = to_categorical(Y, np.max(Y) + 1)  # First class is '0'
        if shuffle_data:
            X, Y = shuffle(X, Y)
    return X, Y


# ==================
#     DATA UTILS
# ==================


def shuffle(*arrs):
    """ shuffle.
    Shuffle given arrays at unison, along first axis.
    Arguments:
        *arrs: Each array to shuffle at unison.
    Returns:
        Tuple of shuffled arrays.
    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


def samplewise_zero_center(X):
    """ samplewise_zero_center.
    Zero center each sample by subtracting it by its mean.
    Arguments:
        X: `array`. The batch of samples to center.
    Returns:
        A numpy array with same shape as input.
    """
    for i in range(len(X)):
        X[i] -= np.mean(X[i], axis=1, keepdims=True)
    return X


def samplewise_std_normalization(X):
    """ samplewise_std_normalization.
    Scale each sample with its standard deviation.
    Arguments:
        X: `array`. The batch of samples to scale.
    Returns:
        A numpy array with same shape as input.
    """
    for i in range(len(X)):
        X[i] /= (np.std(X[i], axis=1, keepdims=True) + _EPSILON)
    return X


def featurewise_zero_center(X, mean=None):
    """ featurewise_zero_center.
    Zero center every sample with specified mean. If not specified, the mean
    is evaluated over all samples.
    Arguments:
        X: `array`. The batch of samples to center.
        mean: `float`. The mean to use for zero centering. If not specified, it
            will be evaluated on provided data.
    Returns:
        A numpy array with same shape as input. Or a tuple (array, mean) if no
        mean value was specified.
    """
    if mean is None:
        mean = np.mean(X, axis=0)
        return X - mean, mean
    else:
        return X - mean


def featurewise_std_normalization(X, std=None):
    """ featurewise_std_normalization.
    Scale each sample by the specified standard deviation. If no std
    specified, std is evaluated over all samples data.
    Arguments:
        X: `array`. The batch of samples to scale.
        std: `float`. The std to use for scaling data. If not specified, it
            will be evaluated over the provided data.
    Returns:
        A numpy array with same shape as input. Or a tuple (array, std) if no
        std value was specified.
    """
    if std is None:
        std = np.std(X, axis=0)
        return X / std, std
    else:
        return X / std


def directory_to_samples(directory, flags=None):
    """ Read a directory, and list all subdirectories files as class sample """
    samples = []
    targets = []
    label = 0
    classes = sorted(os.walk(directory).next()[1])
    for c in classes:
        c_dir = os.path.join(directory, c)
        for sample in os.walk(c_dir).next()[2]:
            if not flags or any(flag in sample for flag in flags):
                samples.append(os.path.join(c_dir, sample))
                targets.append(label)
        label += 1
    return samples, targets
