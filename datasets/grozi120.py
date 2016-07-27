""" GroZi-120 Grocery Products Dataset

Credits: Michele Merler, Carolina Galleguillos, and Serge Belongie.
http://grozi.calit2.net/grozi.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from six.moves import urllib
import zipfile

from tflearn.data_utils import build_image_dataset_from_dir

TRAIN_DATA_URL = "http://grozi.calit2.net/GroZi-120/inVitro.zip"
TEST_DATA_URL = "http://grozi.calit2.net/GroZi-120/inSitu.zip"


def load_data(dirname="datasets/grozi-120-invitro", resize_pics=(256, 256),
              shuffle=True, one_hot=False):
  dataset_file = os.path.join(dirname, 'grozi120.pkl')
  if not os.path.exists(dataset_file):
    maybe_download_and_extract(TRAIN_DATA_URL, dirname)

  X_train, Y_train = build_image_dataset_from_dir(dirname + '/JPEG/',
                                                  dataset_file=dataset_file,
                                                  resize=resize_pics,
                                                  filetypes=['.jpg', '.jpeg'],
                                                  convert_gray=True,
                                                  shuffle_data=shuffle,
                                                  categorical_Y=one_hot)
  # X_test, Y_test =

  # return (X_train, Y_train), (X_test, Y_test)
  return X_train, Y_train


def build_class_directories(base_dir):
  """Reorganizes directory structure of GroZi dataset.

  Data is structured like:
  ```
      BASE_DIR -> INVITRO -> CLASS -> WEB -> SUBTYPE -> IMG_01.(jpg/png)
  ```
  but we want it like:
  ```
      BASE_DIR -> SUBTYPE -> CLASS -> IMG_01.(jpg/png)
  ```
  """
  iv_dir = os.path.join(base_dir, 'inVitro')
  classes = os.walk(iv_dir).next()[1]
  for c in classes:
    c_dir = os.path.join(iv_dir, c)
    web_dir = os.path.join(c_dir, 'web')

    subtypes = os.walk(web_dir).next()[1]
    for s in subtypes:
      s_dir = os.path.join(web_dir, s)
      # pad class number with 0s to match label order
      os.renames(s_dir, os.path.join(base_dir, s, '%03d' % int(c)))
      print("Successfully renamed category %s's directory." % c)


def maybe_download_and_extract(data_url, dest_directory):
  """Downloads and extracts the zip file from website.

  Also reorganizes directory structure
  """
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
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
    zipfile.ZipFile(filepath, 'r').extractall(dest_directory)
    build_class_directories(dest_directory)
