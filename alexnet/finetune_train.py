from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import numpy as np
import tensorflow as tf

from alexnet.finetune import *

FLAGS = tf.app.flags.FLAGS


def main(_):
  run_date_time = datetime.now().strftime("%Y%m%d-%H%M")

  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_graph())

  load_from = (FLAGS.data_split_file
               if FLAGS.data_split_file
               else FLAGS.image_dir)
  xTr, yTr, xTe, yTe = load_data(load_from, date_time=run_date_time)

  class_count = len(xTr)
  if class_count == 0:
    print('No valid folders of images found at ' + load_from)
    return -1
  if class_count == 1:
    print('Only one valid folder of images found at ' + load_from +
          ' - multiple classes are needed for classification.')
    return -1