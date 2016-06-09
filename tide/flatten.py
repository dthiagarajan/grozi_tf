import os
import numpy as np
from PIL import Image

in_dir = '/home/shoffman/Documents/grozi_tf/tide/raw_thumbnails'
num_images = 46


files = [os.path.join(in_dir, 'tide_thumbnail%d.png' % i) for i in range(1,num_images+1)]
out = np.zeros((3073*num_images,1), dtype=np.uint8)
for n, f in enumerate(files):
	im = Image.open(f)
	im = np.array(im, dtype=np.uint8)
	r = im[:,:,0].flatten()
	g = im[:,:,1].flatten()
	b = im[:,:,2].flatten()
	label = [1]
	out[3073*n:3073*(n+1)] = np.array(list(label) + list(r) + list(g) + list(b), dtype=np.uint8)[:, np.newaxis]
out.tofile('tide_train.bin')

files = [os.path.join(in_dir, 'nottide_thumbnail%d.png' % i) for i in range(1,num_images+1)]
out = np.zeros((3073*num_images,1), dtype=np.uint8)
for n, f in enumerate(files):
	im = Image.open(f)
	im = np.array(im, dtype=np.uint8)
	r = im[:,:,0].flatten()
	g = im[:,:,1].flatten()
	b = im[:,:,2].flatten()
	label = [0]
	out[3073*n:3073*(n+1)] = np.array(list(label) + list(r) + list(g) + list(b), dtype=np.uint8)[:, np.newaxis]
out.tofile('nottide_train.bin')

files = [os.path.join(in_dir, 'tide_thumbnail%d.png' % (num_images+i)) for i in range(1,6)]
files = files + [os.path.join(in_dir, 'nottide_thumbnail%d.png' % (num_images+i)) for i in range(1,6)]
out = np.zeros((3073*10,1), dtype=np.uint8)
for n, f in enumerate(files):
	im = Image.open(f)
	im = np.array(im, dtype=np.uint8)
	r = im[:,:,0].flatten()
	g = im[:,:,1].flatten()
	b = im[:,:,2].flatten()
	label = [n < 5]
	out[3073*n:3073*(n+1)] = np.array(list(label) + list(r) + list(g) + list(b), dtype=np.uint8)[:, np.newaxis]
out.tofile('test_batch.bin')
