import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy
import math

from PIL import Image
from keras import backend as K
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract, Add, UpSampling2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

train_input_directory = 'D:/lenless_data/PHD-08/averaged_data/train/'
val_input_directory = 'D:/lenless_data/PHD-08/averaged_data/val/'
test_input_directory = 'D:/lenless_data/PHD-08/averaged_data/test/'

input_image_size = 224
BATCH_SIZE = 4
EPOCHS = 150
random.seed(3)

train_datagen = ImageDataGenerator(rescale=1./255, 
#                                  rotation_range=5,
#                                  width_shift_range=0.1,
#                                  height_shift_range=0.1,
#                                  shear_range=0.2,
#                                  zoom_range=[1.0, 1.3],
#                                  brightness_range = [0.8, 1.5],
#                                  fill_mode='nearest'
                                  )

validation_gen = ImageDataGenerator(rescale=1./255, 
#                                  rotation_range=15,
#                                  width_shift_range=0.1,
#                                  height_shift_range=0.1,
#                                  shear_range=0.2,
#                                  zoom_range=[1.0, 1.3],
#                                  brightness_range = [0.8, 1.5],
#                                  fill_mode='nearest'
                                  )

test_gen = ImageDataGenerator(rescale=1./255, 
#                                  rotation_range=20,
                                  width_shift_range=0.08,
#                                  height_shift_range=0.1,
#                                  shear_range=10,
#                                  zoom_range=[0.94, 1.12],
#                                  brightness_range = [0.30, 1.04],
#                                  fill_mode='nearest'
                                  )

train_generator = train_datagen.flow_from_directory(
        train_input_directory,
        target_size = (input_image_size, input_image_size),
        batch_size = BATCH_SIZE,
        color_mode = 'grayscale',
        class_mode = 'categorical')

validation_generator = validation_gen.flow_from_directory(
        val_input_directory,
        target_size=(input_image_size, input_image_size),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical')

test_generator = test_gen.flow_from_directory(
        test_input_directory,
        target_size=(input_image_size, input_image_size),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical')  

# to check if ImageDataGenerator is ok or not
#img = load_img('C:/Users/my/Desktop/vangoghmuseum-s0089B1991-800.jpg')  # PIL 이미지
#x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
#x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열
#
## 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
## 지정된 `preview/` 폴더에 저장합니다.
#i = 0
#for batch in train_datagen.flow(x, batch_size=1,
#                          save_to_dir='C:/Users/my/Desktop/', save_prefix='cat', save_format='jpg'):
#    i += 1
#    if i > 20:
#        break  # 이미지 20장을 생성하고 마칩니다

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
    predictions = Dense(14, activation='softmax')(x)
    
    lensless_Autoencoder = Model(inputs=input_img, outputs=predictions, name='lensless_AE')
    
    return lensless_Autoencoder

model = lensless_AE()
sgd = SGD(lr=0.001, momentum = 0.9)
model.compile(optimizer=sgd, loss = 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

#model.load_weights('./lensless_classificaiton_phd08_all_data_가~하.h5')

checkpointer = ModelCheckpoint(filepath='./lensless_classificaiton_phd08_all_data_가~하.h5', verbose=1, save_best_only = True)
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
earlystopper = EarlyStopping(monitor='val_loss', patience=20)

hist = model.fit_generator(
        train_generator,
        steps_per_epoch = 331, 
        epochs = 500,
        callbacks = [checkpointer, reduceLR, earlystopper],
        validation_data = validation_generator,
        validation_steps = 215)

model.save_weights('lensless_classificaiton_phd08_1000data_가~하_aug_both.h5')

#print(hist.history['loss'])
#print(hist.history['acc'])
#print(hist.history['val_loss'])
#print(hist.history['val_acc'])

## summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
## summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
## summarize history for lr
plt.plot(hist.history['lr'])
plt.title('model learning rate')
plt.ylabel('lr')
plt.xlabel('epoch')
plt.legend(['lr'], loc='upper left')
plt.show()

print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=215)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(validation_generator.class_indices)
print(output)

#
#test= Image.open('D:/lenless_data/PHD-08/averaged_data/test/파/3.png').convert('L')
#test = np.array(test)   # nparray로
#            
#test = scipy.misc.imresize(test, (input_image_size, input_image_size), 'bicubic')
#
#test = np.expand_dims(test, axis=0)     # input 사이즈에 맞게
#test = np.expand_dims(test, axis=3)
#test = test/255
#
#test = model.predict(test)   # predict
