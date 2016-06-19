from __future__ import division

import os
import numpy as np
from PIL import Image
import resize
num = (resize.s * resize.s * 3) + 1
in_dir = '../tide_photos/raw_thumbnails'
total = 102
num_test = total//10
num_images = total//2 - num_test


files = [os.path.join(in_dir, 'tide_thumbnail%d.png' % i) for i in range(1,num_images+1)]
out = np.zeros((num*num_images,1), dtype=np.uint8)
for n, f in enumerate(files):
	im = Image.open(f)
	im = np.array(im, dtype=np.uint8)
	r = im[:,:,0].flatten()
	g = im[:,:,1].flatten()
	b = im[:,:,2].flatten()
	label = [1]
	out[num*n:num*(n+1)] = np.array(list(label) + list(r) + list(g) + list(b), dtype=np.uint8)[:, np.newaxis]
out.tofile('tide_train.bin')

files = [os.path.join(in_dir, 'nottide_thumbnail%d.png' % i) for i in range(1,num_images+1)]
out = np.zeros((num*num_images,1), dtype=np.uint8)
for n, f in enumerate(files):
	im = Image.open(f)
	im = np.array(im, dtype=np.uint8)
	r = im[:,:,0].flatten()
	g = im[:,:,1].flatten()
	b = im[:,:,2].flatten()
	label = [0]
	out[num*n:num*(n+1)] = np.array(list(label) + list(r) + list(g) + list(b), dtype=np.uint8)[:, np.newaxis]
out.tofile('nottide_train.bin')

files = [os.path.join(in_dir, 'tide_thumbnail%d.png' % (num_images+i)) for i in range(1,num_test+1)]
files = files + [os.path.join(in_dir, 'nottide_thumbnail%d.png' % (num_images+i)) for i in range(1,num_test+1)]
out = np.zeros((num*2*num_test,1), dtype=np.uint8)
for n, f in enumerate(files):
	im = Image.open(f)
	im = np.array(im, dtype=np.uint8)
	r = im[:,:,0].flatten()
	g = im[:,:,1].flatten()
	b = im[:,:,2].flatten()
	label = [n < num_test]
	out[num*n:num*(n+1)] = np.array(list(label) + list(r) + list(g) + list(b), dtype=np.uint8)[:, np.newaxis]
out.tofile('test_batch.bin')
