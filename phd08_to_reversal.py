import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imsave

from PIL import Image
from scipy import ndimage

origin_dir = 'C:/Users/my/Downloads/phd08_png_results/하'
save_dir = 'C:/Users/my/Downloads/phd08_png_BW_changed/하/'
image_size = 28
counter = 0
numbering = 0

        
def reverse_images(directory, save_directory):
    for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    global counter
                    filepath = os.path.join(root, filename) ## 이미지 주소 따기
                    image = ndimage.imread(filepath, mode="L")  ## 이미지 읽기, Grayscale
                    
                    for i in range(0, image_size):
                        for j in range(0, image_size):
                            image[i][j] = 255 - image[i][j]
                    
                    image = Image.fromarray(image)
                    imsave(save_dir + '하' + str(counter) + '.png', image)
                    counter += 1

 
reverse_images(origin_dir, save_dir)
                
#plt.imshow(image)
