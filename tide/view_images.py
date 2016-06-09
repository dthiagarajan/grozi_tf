import os
import binascii
import numpy as np
from PIL import Image

# data_dir = "../tide"

# grozi = open(data_dir+"/tide_train.bin","rb")
grozi = open('test.bin', 'rb')


for i in range(10):
	grozi_label = grozi.read(1)
	print (binascii.hexlify(grozi_label))
	grozi_image = grozi.read(3072)
	grozi_image = np.frombuffer(grozi_image,dtype=np.uint8)
	grozi_image = grozi_image.reshape([3,1024]).T.reshape([32,32,3])
	grozi_im = Image.fromarray(grozi_image,mode='RGB')
	grozi_im.save(open('grozi%d.jpg' % i,'w'))
