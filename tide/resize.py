import os, sys
from PIL import Image
size = 32, 32

def init(s):
	return 'video' + str(s) + '.png'
files = map(init, range(1,49))
for f in files:
	out = f.split('.')[0] + '_mod.png'
	try:
		im = Image.open(f)
		im.thumbnail(size, Image.ANTIALIAS)
		im = im.resize((32,32), Image.ANTIALIAS)
		im.save(out, "PNG")
	except IOError:
		print "cannot create thumbnail for '%s'" % f
