import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy

from scipy.misc import imsave
from PIL import Image
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, MaxPooling2D

input_image_size = 224

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)  
    return out

def lensless_AE(): 
    
    input_shape = (None, None, 1) #(None)
    
    input_img = Input(shape=input_shape)
    
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    #x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    #x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    
    lensless_Autoencoder = Model(inputs=input_img, outputs=x, name='lensless_AE')
    
    return lensless_Autoencoder

model = lensless_AE()

model.load_weights('./lensless_AE_ver1.h5')

test= Image.open('C:/Users/my/Desktop/image_SR_for_대열이형/1.png').convert('L')
test = np.array(test)   # nparray로
            
test = scipy.misc.imresize(test, (input_image_size, input_image_size), 'bicubic')

test = np.expand_dims(test, axis=0)     # input 사이즈에 맞게
test = np.expand_dims(test, axis=3)
test = im2double(test)

test = model.predict(test)   # predict

test = np.squeeze(test, axis=0)
test = np.squeeze(test, axis=2)

test = Image.fromarray(np.uint8(test))

plt.imshow(test)

imsave('C:/Users/my/Desktop/image_SR_for_대열이형/11.png', test)





