import os, sys
from random import choice
from PIL import Image

situ_dir  = '../tide_photos/inSitu/%d/video'
vitro_dir = '../tide_photos/inVitro/%d/web/JPEG'
out_dir   = '../tide_photos/raw_thumbnails'
s = 448
size = (448, 448)
nottide  = range(1,121)
nottide.remove(34)


files = [os.path.join(situ_dir % choice(nottide), 'video1.png') for i in range(48)]
files = files + [os.path.join(vitro_dir % choice(nottide), 'web1.jpg') for i in range(3)]

for n, f in enumerate(files, start=1):
	out = os.path.join(out_dir, 'nottide_thumbnail%d.png' % n)
	try:
		im = Image.open(f)
		im = im.resize(size, Image.ANTIALIAS)
		im.save(out, "PNG")
	except IOError:
		print "cannot create thumbnail for '%s'" % f

files = [os.path.join(situ_dir % 34, 'video%d.png' % i) for i in range(1,49)]
files = files + [os.path.join(vitro_dir % 34, 'web%d.jpg' % i) for i in range(1,4)]

for n, f in enumerate(files, start=1):
	out = os.path.join(out_dir, 'tide_thumbnail%d.png' % n)
	try:
		im = Image.open(f)
		im = im.resize(size, Image.ANTIALIAS)
		im.save(out, "PNG")
	except IOError:
		print "cannot create thumbnail for '%s'" % f
