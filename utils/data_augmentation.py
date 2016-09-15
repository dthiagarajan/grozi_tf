"""Real-time data augmentation techniques.

These transformations should be applied to Tensors.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tflearn import ImageAugmentation


class ImageAugmentationWithRandomBackground(ImageAugmentation):
  def __init__(self):
    super(ImageAugmentationWithRandomBackground, self).__init__()

  def add_random_background(self):
    self.methods.append(self._random_background)
    self.args.append(None)

  def _random_background(self, batch):
    for i in range(len(batch)):
      shape = batch[i].shape
      assert shape[2] == 4, 'Input must be a .png file for random background' \
                            'augmentations.'
      background = np.random.randint(256, size=shape, dtype='float32')
      background[:,:,3] = 255.
      where = np.where(np.isclose(batch[i][:,:,3], 0))[:2]
      batch[i][where] = background[where]
