from PIL import Image
import numpy as np

def init(s):
        return 'video' + str(s) + '_mod.png'
files = map(init, range(1,49))
images = []
for f in files:
	im = Image.open(f)
	im = (np.array(im))
	r = im[:,:,0].flatten()
	g = im[:,:,1].flatten()
	b = im[:,:,2].flatten()
	label = [0]
	images.append(np.array(list(label) + list(r) + list(g) + list(b)))
out = np.concatenate((images[0],images[1]))
for i in range(2,len(images)):
	out = np.concatenate((out,images[i]))
out.tofile('tide.bin')


