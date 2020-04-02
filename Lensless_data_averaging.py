import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imsave

from PIL import Image
from scipy import ndimage

directory = 'D:/lenless_data/train_Cap/3_'
save_dir = 'D:/lenless_data/averaged_data/train/3/'
image_size = 600
num_per_image = 20
counter = 0
numbering = 0

def averaging_images(tempImages):
    global numbering
    averaged_image = np.zeros((image_size, image_size))
    
    for i in range(0, num_per_image):
        averaged_image += tempImages[i]
        
    averaged_image /= num_per_image
    averaged_image = Image.fromarray(np.uint8(averaged_image))
    
    imsave(save_dir + str(numbering) + '.png', averaged_image)
    numbering += 1

        
def average_inputimages(directory, tempImages):
    for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    global counter
                    filepath = os.path.join(root, filename) ## 이미지 주소 따기
                    image = ndimage.imread(filepath, mode="L")  ## 이미지 읽기, Grayscale
                    counter += 1;
                    tempImages.append(image)
                    
                    if ((counter%num_per_image) == 0) :
                        averaging_images(tempImages)   ## 평균
                        counter = 0
                        tempImages = []

tempImages = []
average_inputimages(directory, tempImages)