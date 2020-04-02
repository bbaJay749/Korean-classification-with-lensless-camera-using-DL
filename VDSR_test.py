from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, BatchNormalization, add
from keras.preprocessing import image
from scipy.misc import imsave
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from PIL import Image
import os
import re

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)  
    return out

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

V = 512    # 목표하는 해상도
H = 512
input_shape = (V, H, 1)

networkDepth = 38 ##20
input_img = Input(shape=input_shape)

x = BatchNormalization()(input_img)

x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
x = Activation('relu')(x)

for i in range(1, networkDepth-1, 2):
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Activation('relu')(x)
    
x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(x)    

model = Model(inputs=input_img, outputs=x, name='VDSR')

model.load_weights('./vdsr_x4_noise_1.h5')

folder = 'C:/Users/my/Desktop/SR_for_대열이형/x4_nois2.png'

for root, dirnames, filenames in os.walk(folder):
    for filename in filenames:
        if re.search("\.(JPG|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename) ## 이미지 주소 따기
            Iycbcr = Image.open(filepath).convert('YCbCr')
            
            Iy, Icb, Icr = Iycbcr.split()   
            
            Iy_bicubic = np.array(Iy)   # nparray로
            Icb_bicubic = np.array(Icb)
            Icr_bicubic = np.array(Icr)
            
            Iy_bicubic = np.expand_dims(Iy_bicubic, axis=0)     # input 사이즈에 맞게
            Iy_bicubic = np.expand_dims(Iy_bicubic, axis=3)
            Iy_bicubic = im2double(Iy_bicubic)
            
            Iresidual = model.predict(Iy_bicubic)   # predict
            
            Iy_bicubic = np.reshape(Iy_bicubic, (V, H)) 
            Iresidual = np.reshape(Iresidual, (V, H))
            
            Isr = Iy_bicubic + Iresidual    # 예측한 값과 원래 값을 더해서 사용

            Ivdsr = np.zeros((V, H, 3))
            
            Ivdsr[:, :, 0] = Isr * 255
            Ivdsr[:, :, 1] = Icb_bicubic 
            Ivdsr[:, :, 2] = Icr_bicubic    # 다시 I, Cb, Cr을 합친다.
            
            Ivdsr = ycbcr2rgb(Ivdsr)
            
            Ivdsr = Image.fromarray(Ivdsr.astype('uint8'))
            
            plt.imshow(Ivdsr)
            
            imsave(filepath + '.png', Ivdsr)


## 아래 코드 실행 시 N X N light field data가 모두 spatially SR 됨. Angularly X 2 될 것을 감안해 numbering 됨        
        
index_counter = 1
angular_resolution = 7
OG_counter = 1

for i in range(1, pow(angular_resolution, 2) + 1):
            global index_counter
            global OG_counter
            
            Iycbcr= Image.open('D:/Dropbox (3dlab Inha)/병아리/Lab_meeting/박대열/x4/dncnn_rabbit_1_1_'+ str(index_counter) +'.png.png').convert('YCbCr')
            OG_counter += 1
            
           # Iycbcr= Image.open('C:/Users/my/Desktop/dncnn_rabbit_1_19_16.png.png').convert('YCbCr')
            
            Iy, Icb, Icr = Iycbcr.split()       # I, Cb, Cr로 분리
           
#            Iy_bicubic = Iy.resize((round(H/2), round(V/2)), Image.BICUBIC)
#            Icb_bicubic = Icb.resize((round(H/2), round(V/2)), Image.BICUBIC)
#            Icr_bicubic = Icr.resize((round(H/2), round(V/2)), Image.BICUBIC)
#            
#            Iy_bicubic = Iy_bicubic.resize((H, V), Image.BICUBIC)   # 원하는 해상도로 BICUBIC resize
#            Icb_bicubic = Icb_bicubic.resize((H, V), Image.BICUBIC)
#            Icr_bicubic = Icr_bicubic.resize((H, V), Image.BICUBIC)
            
            Iy_bicubic = np.array(Iy_bicubic)   # nparray로
            Icb_bicubic = np.array(Icb_bicubic)
            Icr_bicubic = np.array(Icr_bicubic)
            
            Iy_bicubic = np.expand_dims(Iy_bicubic, axis=0)     # input 사이즈에 맞게
            Iy_bicubic = np.expand_dims(Iy_bicubic, axis=3)
            Iy_bicubic = im2double(Iy_bicubic)
            
            Iresidual = model.predict(Iy_bicubic)   # predict
            
            Iy_bicubic = np.reshape(Iy_bicubic, (V, H)) 
            Iresidual = np.reshape(Iresidual, (V, H))
            
            Isr = Iy_bicubic + Iresidual    # 예측한 값과 원래 값을 더해서 사용

            Ivdsr = np.zeros((V, H, 3))
            
            Ivdsr[:, :, 0] = Isr * 255
            Ivdsr[:, :, 1] = Icb_bicubic 
            Ivdsr[:, :, 2] = Icr_bicubic    # 다시 I, Cb, Cr을 합친다.
            
            Ivdsr = ycbcr2rgb(Ivdsr)
            
            Ivdsr = Image.fromarray(Ivdsr.astype('uint8'))
            
            plt.imshow(Ivdsr)
            
            #imsave('C:/Users/my/Desktop/dncnn_rabbit_1_19_161.png.png' , Ivdsr)
        
            imsave('D:/Dropbox (3dlab Inha)/병아리/Lab_meeting/박대열/x4_SR/dncnn_rabbit_1_1_' + str(index_counter) + '.png' , Ivdsr)
        
            index_counter += 1

## 단순히 Bicubic Resize로 이미지들의 사이즈를 줄이는 코드
                
index_counter = 0
angular_resolution = 9
OG_counter = 1

for i in range(1, pow(angular_resolution, 2) + 1):
            global index_counter
            index_counter += 1
            Iycbcr= Image.open('C:/Users/my/Desktop/LF_SR_TEMP/bunny/bunny (' + str(index_counter) + ').png')
            
            Iycbcr = Iycbcr.resize((256, 256), Image.BICUBIC)
            
            imsave('C:/Users/my/Desktop/LF_SR_TEMP/bunny_resized/bunny_'+ str(index_counter) + '.png' , Iycbcr)
