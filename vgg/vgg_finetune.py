# -*- coding: utf-8 -*-
"""Fine-tuning with VGG-16 architecture pre-trained on ImageNet data.

All weights are restored except last layer (softmax) that will be retrained
to match the new task (fine-tuning).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from six.moves import urllib

import tflearn

from datasets import grozi120
from vgg.vgg16 import vgg16

DATA_URL = 'https://www.dropbox.com/s/9li9mi4105jf45v/vgg16.tflearn?dl=1'


def retrain(output_filename):
  num_classes = 120

  # Real-time data preprocessing
  img_prep = tflearn.ImagePreprocessing()
  img_prep.add_featurewise_zero_center()
  img_prep.add_featurewise_stdnorm()

  # Real-time data augmentation
  img_aug = tflearn.ImageAugmentation()
  img_aug.add_random_blur(sigma_max=5.)
  img_aug.add_random_crop((224, 224))
  img_aug.add_random_rotation(max_angle=25.)

  softmax = vgg16(softmax_size=num_classes, restore_softmax=False,
                  data_preprocessing=img_prep, data_augmentation=img_aug)
  regression = tflearn.regression(softmax, optimizer='rmsprop',
                                  loss='categorical_crossentropy',
                                  learning_rate=0.001)

  model = tflearn.DNN(regression, checkpoint_path=output_filename,
                      max_checkpoints=3, tensorboard_verbose=3)
  # Load pre-existing model, restoring all weights, except softmax layer ones
  model_file = 'vgg/vgg16.tflearn'
  if not os.path.exists(model_file):
    maybe_download(DATA_URL, 'vgg')
  model.load(model_file)

  # Start fine-tuning
  X, Y = grozi120.load_data()
  model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
            show_metric=True, batch_size=64, snapshot_step=200,
            snapshot_epoch=False, run_id=output_filename)

  model.save(output_filename)


def maybe_download(data_url, dest_directory):
  """Downloads pre-trained model file"""
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filename = filename.split('?')[0]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
