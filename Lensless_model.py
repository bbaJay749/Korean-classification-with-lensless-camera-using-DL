import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import scipy
import math

from PIL import Image
from keras import backend as K
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract, Add, UpSampling2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

input_directory = 'D:/lenless_data/averaged_data/test/'
output_directory = 'D:/Dropbox (3dlab Inha)/박재용/Projects/정보통신종합설계/mnist-normal/test/'
test_input_directory = 'D:/lenless_data/averaged_data/train/'
input_image_size = 224
BATCH_SIZE = 4
EPOCHS = 150
n_class = 4

output_list = os.listdir(test_input_directory)

def load_outputlabels(directory, label_list):
    for dirname in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, dirname)):
            label_list.append(int(dirname))
    label_list = to_categorical(label_list, n_class)

def load_outputimages(directory, label_list):
    for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    filepath = os.path.join(root, filename) ## 이미지 주소 따기
                    image = ndimage.imread(filepath, mode="L")  ## 이미지 읽기, Grayscale
                    image = image.astype('float32') / 255.0 
                    label_list.append(image)
                    
def load_inputimages(directory, label_list):
    for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    filepath = os.path.join(root, filename) ## 이미지 주소 따기
                    image = ndimage.imread(filepath, mode="L")  ## 이미지 읽기, Grayscale
                    image = scipy.misc.imresize(image, (input_image_size, input_image_size), 'bicubic')
                    image = image.astype('float32') / 255.0 
                    label_list.append(image)

inputImages = []
outputImages = []
outputlabels = []
test_inputImages = []
test_outputlabels = []
    
load_inputimages(input_directory, inputImages)
load_outputimages(output_directory, outputImages)
load_outputlabels(input_directory, outputlabels)

load_inputimages(test_input_directory, test_inputImages)
load_outputlabels(test_input_directory, test_outputlabels)

inputImages = np.array(inputImages)
array_shape = np.append(inputImages.shape[0:3], 1)
inputImages = np.reshape(inputImages, (array_shape))    

outputImages = np.array(outputImages)
array_shape = np.append(outputImages.shape[0:3], 1)
outputImages = np.reshape(outputImages, (array_shape)) 

test_inputImages = np.array(test_inputImages)
array_shape = np.append(test_inputImages.shape[0:3], 1)
test_inputImages = np.reshape(test_inputImages, (array_shape))   

#################################################################################################

def lensless_AE(): 
    
    input_shape = (input_image_size, input_image_size, 1) #(None)
    
    input_img = Input(shape=input_shape)
    
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    for i in range(3):
        x = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(2048, activation = 'relu' )(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)
    
    lensless_Autoencoder = Model(inputs=input_img, outputs=predictions, name='lensless_AE')
    
    return lensless_Autoencoder

model = lensless_AE()
sgd = SGD(lr=0.001)
model.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.load_weights('./lensless_AE_ver2.h5')

checkpointer = ModelCheckpoint(filepath='lensless_AE_ver2.h5', verbose=1, save_best_only = True)
earlyStopper = EarlyStopping(monitor='val_cost', min_delta=0, patience=20, verbose=1, mode='auto')

model.fit(x=inputImages, y=label_list, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
          verbose=1, callbacks = [checkpointer, earlyStopper], validation_split = 0.2)

model.save('lensless_AE_ver1.h5')  # creates a HDF5 file 

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(test_inputImages, test_outputlabels, batch_size=4)
print('test loss, test acc:', results)
