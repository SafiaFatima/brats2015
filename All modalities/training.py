
# -*- coding: utf-8 -*-
"""

@author: SAFIA FATIMA
"""
from __future__ import print_function
import model as mm

from sklearn.utils import class_weight
import numpy as np
from matplotlib import pyplot as plt
import gc
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from keras.initializers import RandomNormal
import tensorflow as tf
import keras
from keras.models import Sequential, Model,  model_from_json
from keras.layers import Dropout, Activation, Reshape
from keras.layers import Input, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, add
from keras.layers import Input, merge, Convolution2D
from keras.initializers import constant
from keras.layers import Dense, Flatten, Reshape
from keras.losses import categorical_crossentropy
from keras.layers.advanced_activations import PReLU

from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop, Nadam, Adagrad

from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from data_prep_noaug import load_train_data, load_val_data, load_test_data

from sklearn.metrics import classification_report,confusion_matrix


# an argument to your Session's config arguments, it might help a little, 
# though it won't release memory, it just allows growth at the cost of some memory efficiency
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.80
sess = tf.Session(config=config)
K.set_session(sess)
# with tf.Session(config = config) as s:
seed = 7
# ==========================================================================
smooth =  1.0
nclasses = 5 # no of classes, if the output layer is softmax
# nclasses = 1 # if the output layer is sigmoid
img_rows = 240
img_cols = 240
#img_rows = 160
#img_cols = 160


def dice_coe(output, target, axis=1, smooth=1):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.
    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__
    """
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


def dice_coef_loss(y_true, y_pred):
    
    #return 1-dice_hard_coe(y_true, y_pred)

    return 1-dice_coe(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
    return loss

def cnnBRATsInit_unet_5():  
    # using LeakyReLU activation
    
    # the output layer is softmax, loss should be categorical_crossentropy
    # Downsampling 2^4 = 16 ==> 240/16 = 15
    
    
    inputs = Input((img_rows, img_cols, 4))
    # Normalize all data before extracting features
    input_nor = BatchNormalization()(inputs)
    
    # Block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(input_nor)
    conv1 = BatchNormalization()(conv1)
    #conv1 = Activation('relu')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    #conv1 = Activation('relu')(conv1)    
    conv1 = LeakyReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    
    pool1 = Dropout(0.3)(pool1)

    # Block 2
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(pool1) 
    conv2 = BatchNormalization()(conv2)
    #conv2 = Activation('relu')(conv2)    
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    #conv2 = Activation('relu')(conv2)    
    conv2 = LeakyReLU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.3)(pool2)
    
    # this MaxPooling2D has the same result with the above line.
    # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    # W_new = (W - F + 2P)/S + 1
    # W_new = (120 - 3 + 2*1)/2 + 1 = 60.5 --> 60 (round down)
    # P = 1, because the padding parameter is 'same', the size of W has no change
    # before divide by Stride parameter
    
   
    # Block 3
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    #conv3 = Activation('relu')(conv3)    
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    #conv3 = Activation('relu')(conv3)   
    conv3 = LeakyReLU()(conv3)    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.3)(pool3)
    #dropout1 = Dropout(0.9)(pool3)
    
    # Block 4
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    #conv4 = Activation('relu')(conv4)    
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    #conv4 = Activation('relu')(conv4)    
    conv4 = LeakyReLU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.3)(pool4)
    # dropout2 = Dropout(0.5)(pool4)
    
    # Block 5
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    #conv5 = Activation('relu')(conv5)    
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    #conv5 = Activation('relu')(conv5)   
    conv5 = LeakyReLU()(conv5)
    conv5 = Dropout(0.3)(conv5)    
    # Block 6
    # axis = -1 same as axis = last dim order, in this case axis = 3
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), 
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(up6) 
    conv6 = BatchNormalization()(conv6)
    #conv6 = Activation('relu')(conv6)    
    conv6 = LeakyReLU()(conv6)
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    #conv6 = Activation('relu')(conv6)    
    conv6 = LeakyReLU()(conv6)
    conv6 = Dropout(0.3)(conv6)
    # Block 7
    up7 = concatenate([Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), 
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(up7) 
    conv7 = BatchNormalization()(conv7)
    #conv7 = Activation('relu')(conv7)    
    conv7 = LeakyReLU()(conv7)    
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    #conv7 = Activation('relu')(conv7)    
    conv7 = LeakyReLU()(conv7)    
    #conv7 = Dropout(0.5)(conv7)
    # Block 8
    up8 = concatenate([Conv2DTranspose(128, kernel_size=(3,3), strides=(2, 2), 
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    #conv8 = Activation('relu')(conv8)    
    conv8 = LeakyReLU()(conv8)    
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    #conv8 = Activation('relu')(conv8)    
    conv8 = LeakyReLU()(conv8)    
    #conv8 = Dropout(0.5)(conv8)
    # Block 9
    up9 = concatenate([Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), 
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(up9) 
    conv9 = BatchNormalization()(conv9)
    #conv9 = Activation('relu')(conv9)    
    conv9 = LeakyReLU()(conv9)   
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    #conv9 = Activation('relu')(conv9)   
    conv9 = LeakyReLU()(conv9)   
    #conv9 = Dropout(0.5)(conv9)
    
    # Block 10
    conv10 = Conv2D(nclasses, kernel_size=(1, 1), padding='same')(conv9)
    
    # Block 10
    # curr_channels = nclasses
    _, curr_width, curr_height, curr_channels = conv10._keras_shape
    out_conv10 = Reshape((curr_width * curr_height, curr_channels))(conv10)
    act_soft = Activation('softmax')(out_conv10)
    
    model = Model(inputs=[inputs], outputs=[act_soft])
    
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam,loss=dice_coef_loss,metrics=[dice_coe])
    
    return model



#for saving weights for testing the model later
def save_trained_agu_model(model):   
    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_BRATs_unet_new.json", "w") as json_file:    
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_BRATs_unet_new.h5")    
    print("Saved model to disk")
    
        
#call this function to display each slice one by one
def show_2Dimg(img_1, img_2):
    fig, axes = plt.subplots(ncols=2)
    ax = axes.ravel()
    ax[0].imshow(img_1, cmap=plt.cm.gray)
    ax[1].imshow(img_2, cmap=plt.cm.gray)
    plt.show()
    


#main fuction
def train_network_aug():
    # Apply a set of data augmentation methods
    print('Loading training data...')
    imgs_train, imgs_label_train = load_train_data('HGG')
    print('Imgs train shape', imgs_train.shape)
    print('Imgs label shape', imgs_label_train.shape)
    
    print('Loading validation data...')
    imgs_val, imgs_label_val = load_val_data()
    print('Imgs validation shape', imgs_val.shape)
    print('Imgs validation label shape', imgs_label_val.shape)
             
    print('Augmenting the training and validation data...')
    imgs_train = imgs_train.astype('float32') 
  
    imgs_val = imgs_val.astype('float32')

    batch_size = 12
    datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
    
    # fit parameters from data
    datagen.fit(imgs_train)     
    train_generator = datagen.flow(imgs_train, imgs_label_train, batch_size=batch_size)
    

        
    # this is the augmentation configuration we will use for testing:
    val_datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)

    # fit parameters from data
    val_datagen.fit(imgs_val) 
    
    # val_datagen.fit(imgs_train)   
    val_generator = val_datagen.flow(imgs_val, imgs_label_val, batch_size=batch_size)
    # datagen.standardize(imgs_val)
    adam = Adam(lr=0.0001)
    
    print('Creating and compiling model...')    
    model = cnnBRATsInit_unet_5()
    #model = mm.build(240,240,5)
    #model.compile(optimizer=adam,loss=dice_coef_loss,metrics=[dice_coe])
    model.summary()
    model_checkpoint = ModelCheckpoint('weights.h5',monitor='val_dice_coef_loss',save_best_only=True)
    
    print('Fitting model with the data augmentation ...')    
    history = model.fit_generator(train_generator,                        
                                  steps_per_epoch=(imgs_train.shape[0] // batch_size) + 1,                        
                                  epochs=100,                        
                                  verbose=1,
                                  callbacks=[model_checkpoint],
                                  validation_data=val_generator,
                                  validation_steps=(imgs_val.shape[0] // batch_size) + 1,
                    		  use_multiprocessing=True,
                                  workers=6)
    
    print('Predicting model with validation data...')
    prop = model.predict_generator(val_generator, steps=(imgs_val.shape[0] // batch_size) + 1,verbose=1)    
    #prop = model.predict(imgs_val, batch_size=batch_size, verbose=1)
    print(prop.shape)
    save_trained_agu_model(model)  
    
    
    # release memory in GPU and RAM
    del history
    del model
    for i in range(15):
        gc.collect()
        
if __name__ == '__main__':
    
    train_network_aug()   
