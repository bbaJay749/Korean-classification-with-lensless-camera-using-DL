import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imsave
import random

from PIL import Image
from scipy import ndimage

folder_list = ['train', 'test', 'val']
char_list = ['가', '나', '다', '라', '마', '바', '사', '아', '자', '차', '카', '타', '파', '하']
root_directory = 'D:/lenless_data/PHD-08/averaged_data/'
directory = ''
save_dir = ''
image_size = 600
crop_portion = 0.75
crop_image_size = int(image_size * crop_portion)
numbering = 0
varities = 50
        
def crop_inputimages(f_directory, f_save_dir, f_numbering):
    for root, dirnames, filenames in os.walk(f_directory):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    filepath = os.path.join(root, filename) ## 이미지 주소 따기
                    image = ndimage.imread(filepath, mode="L")  ## 이미지 읽기, Grayscale
                    for t in range(0, varities):
                        H_shift = random.randint(1, int(image_size * (1 - crop_portion) - 1))
                        V_shift = random.randint(1, int(image_size * (1 - crop_portion) - 1))
                        crop_image = image[H_shift : H_shift + crop_image_size, V_shift : V_shift + crop_image_size]
                        imsave(f_save_dir + 'shift' + str(f_numbering) + '.png', crop_image)
                        f_numbering += 1

for m in range(0, len(folder_list)):
    for k in range(0, len(char_list)):
        directory = root_directory + str(folder_list[m]) + '/' + str(char_list[k]) + '/'
        save_dir = root_directory + str(folder_list[m]) + '/shift/' + str(char_list[k]) + '_shift/'
        numbering = 0
        crop_inputimages(directory, save_dir, numbering)
