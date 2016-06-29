## Blurring training data to do a bit of data augmentation
## The majority of the augmentation will be done using tflearn's built-in functionality

import tensorflow as tf
import math, os
from PIL import Image

# Randomly distorts an image.
def randomly_distort(image,index,c):
  img = Image.open(image)
  width,height = img.size
  img_data = img.load()          
  output = Image.new('RGB',img.size,"gray")  
  output_img = output.load()    
  pix = [0, 0]
  delta_x = 40     #you can lower the delta for high distortion
  delta_y = 90     #or you can higher the delta for low distortion

  for x in range(width):
    for y in range(height):
        x_shift, y_shift =  ( int(abs(math.sin(x)*width/delta_x)) ,
                                int(abs(math.tan(math.sin(y)))*height/delta_y))
        if x + x_shift < width:
          pix[0] = x + x_shift
        else:
          pix[0] = x
        
        if y + y_shift < height :
          pix[1] = y + y_shift
        else:
          pix[1] = y
        output_img[x,y] = img_data[tuple(pix)]
  output.save('../train_photos/' + str(index) + '/video' + str(c) + '.png')

# Augments the directory with distorted images for training.
def augment_dir(d, index):
  print 'working on ' + d
  count = 1 + len(os.listdir(d))
  for fn in os.listdir(d):
    randomly_distort(d+fn,index,count)
    count += 1

for i in range(0,120):
  d = "../train_photos/" + str(i) + "/"
  augment_dir(d,i)




