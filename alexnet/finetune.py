"""Performs 'fine-tuning' of a pre-trained network (AlexNet in this case) on the
dataset at hand.

The model used should be trained on ImageNet or equivalent large-scale dataset
for broad applicability (see https://arxiv.org/abs/1310.1531 for more details).

For AlexNet, the last layer (fc7) is a 4096-dimensional vector. This is used to
train a new softmax layer for the N classes in the dataset at hand.

This program outputs a TensorFlow graph and label mapping to be used at
test-time.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import pickle

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

import utils.data_augmentation as aug
from utils.data_utils import build_image_dataset_from_dir

FLAGS = tf.app.flags.FLAGS

# Input and output file flags.
tf.app.flags.DEFINE_string(
    'image_dir', '~/Documents/grozi_tf/train_photos/',
    """Path to folders of labeled images."""
)
tf.app.flags.DEFINE_string(
    'output_dir', '~/Documents/grozi_tf/graphs/',
    """Where to save the output graph and label mapping."""
)
tf.app.flags.DEFINE_string(
    'data_split_file', '',
    """Optional. File to load training/testing split data from (.npz)."""
    """If used, overrides 'image_dir' flag."""
)

# Details of the training configuration.
tf.app.flags.DEFINE_integer(
    'how_many_training_steps', 4000,
    """How many training steps to run before ending."""
)
tf.app.flags.DEFINE_float(
    'learning_rate', 0.01,
    """How large a learning rate to use when training."""
)
tf.app.flags.DEFINE_integer(
    'testing_percentage', 10,
    """What percentage of images to use as a test set."""
)
tf.app.flags.DEFINE_integer(
    'validation_percentage', 10,
    """What percentage of images to use as a validation set."""
)
tf.app.flags.DEFINE_integer(
    'eval_step_interval', 10,
    """How often to evaluate the training results."""
)
tf.app.flags.DEFINE_integer(
    'train_batch_size', 100,
    """How many images to train on at a time."""
)
tf.app.flags.DEFINE_integer(
    'test_batch_size', 500,
    """How many images to test on at a time. This test set is only used"""
    """infrequently to verify the overall accuracy of the model."""
)
tf.app.flags.DEFINE_integer(
    'validation_batch_size', 100,
    """How many images to use in an evaluation batch. This validation set is"""
    """ used much more often than the test set, and is an early indicator of"""
    """ how accurate the model is during training."""
)

# File-system cache locations.
tf.app.flags.DEFINE_string(
    'model_dir', '/home/shoffman/Documents/image_retraining/imagenet/',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt."""
)
# tf.app.flags.DEFINE_string(
#     'bottleneck_dir', '/home/shoffman/Documents/image_retraining/bottleneck/',
#     """Path to cache bottleneck layer values as files."""
# )
tf.app.flags.DEFINE_string(
    'final_tensor_name', 'retrained_softmax',
    """The name of the output classification layer in the retrained graph."""
)

# Controls the distortions used during training.
# tf.app.flags.DEFINE_boolean(
#     'flip_left_right', False,
#     """Whether to randomly flip half of the training images horizontally."""
# )
# tf.app.flags.DEFINE_integer(
#     'random_crop', 0,
#     """A percentage determining how much of a margin to randomly crop off the"""
#     """ training images."""
# )
tf.app.flags.DEFINE_integer(
    'random_scale', 0,
    """A percentage determining how much to randomly scale up the size of the"""
    """ training images by."""
)
# tf.app.flags.DEFINE_integer(
#     'random_brightness', 0,
#     """A percentage determining how much to randomly multiply the training"""
#     """ image input pixels up or down by."""
# )

ACCEPTED_FILETYPES = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']

# AlexNet-specific information.
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 4096
MODEL_INPUT_WIDTH = 224
MODEL_INPUT_HEIGHT = 224
MODEL_INPUT_DEPTH = 3


def load_data(load_from, resize=(256, 256), shuffle=True,
              one_hot=False, save_out=True, date_time=''):
  """Loads data from the specified location and splits it into training and
  testing sets.

  Data is optionally resized, shuffled, and/or normalized.

  Args:
    load_from: If this is a file, loads cached data from file. Otherwise, looks
    through directory structure to determine classes and instances
    resize: tuple of ints (new width, new height). Size of output images
    shuffle: should data be shuffled?
    one_hot: return labels as one-hot vectors?
    save_out: cache data after loading?
    date_time: date-time string to append to filename if saving

  Returns:
    xTr: list of training image paths per class (list of lists)
    yTr: list of training labels per class corresponding to xTr
    xTe: list of testing image paths
    yTe: list of testing labels
  """
  if os.path.isfile(load_from):
    with open(load_from, 'rb') as infile:
      xTr, yTr, xTe, yTe = pickle.load(infile)
  else:
    xTr, yTr, xTe, yTe = (
        build_image_dataset_from_dir(load_from,
                                     resize=resize,
                                     filetypes=ACCEPTED_FILETYPES,
                                     convert_gray=False,
                                     shuffle_data=shuffle,
                                     categorical_Y=one_hot)
    )
    if save_out:
      out_name = os.path.join(FLAGS.output_dir, 'image_data_' + date_time)
      with open(out_name, 'wb') as outfile:
        pickle.dump([xTr, yTr, xTe, yTe], outfile)
  return xTr, yTr, xTe, yTe


def create_graph(model_filename, return_elements=None):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

    Args:
      model_filename: path to graph def file (.pb)
      return_elements: list of strings containing op or tensor names from graph

    Returns:
      Graph holding the trained network, and various tensors we'll be
      manipulating.
    """
  with tf.Session() as sess:
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      # don't add a prefix to the names in graph_def (name='')
      returned_elements = tf.import_graph_def(graph_def, name='',
                                              return_elements=return_elements)
  if returned_elements is not None:
    return [sess.graph] + returned_elements
  else:
    return sess.graph


def add_input_distortions(rotation, crop, zoom, color, background):
  pass
