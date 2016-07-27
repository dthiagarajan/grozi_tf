"""Run full retraining procedure using VGG-16 architecture."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

from vgg.vgg_finetune import retrain


start_time = datetime.now().strftime("%Y%m%d-%H%M")
retrain('vgg16_finetuned_' + start_time)
