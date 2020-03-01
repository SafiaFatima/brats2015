# -*- coding: utf-8 -*-
"""
@author: SAFIA FATIMA
"""
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import gc
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, Activation, Reshape
from keras.layers import Input, Conv2DTranspose
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D, concatenate
from keras.layers import Input, merge, Convolution2D
from keras.initializers import constant
from keras.layers import Dense, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from F_data_prep1 import load_train_data, load_val_data
from sklearn.metrics import classification_report,confusion_matrix
from keras.losses import categorical_crossentropy

# an argument to your Session's config arguments, it might help a little, 
# though it won't release memory, it just allows growth at the cost of some memory efficiency
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
#config.intra_op_parallelism_threads=4
#config.inter_op_parallelism_threads=2
#config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess = tf.Session(config=config)
K.set_session(sess)
# with tf.Session(config = config) as s:
seed = 7
# ==========================================================================
smooth =  50
nclasses = 5 # no of classes, if the output layer is softmax
# nclasses = 1 # if the output layer is sigmoid
img_rows = 240
img_cols = 240
def step_decay(epochs):
    init_rate = 0.003
    fin_rate = 0.00003
    total_epochs = 24
    print ('ep: {}'.format(epochs))
    if epochs<25:
        lrate = init_rate - (init_rate - fin_rate)/total_epochs * float(epochs)
    else: lrate = 0.00003
    print ('lrate: {}'.format(model.optimizer.lr.get_value()))
    return lrate

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
            - If either output or target are empty (all pixels are background), dice = ```smooth/(sma$
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



def bce_dice_loss(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
    return loss


def dice_coef_loss(y_true, y_pred):
    
    return 1-dice_coe(y_true, y_pred)


def cnnBRATsInit_unet_5():  
    
    
    # the output layer is softmax, loss should be categorical_crossentropy
    # Downsampling 2^4 = 16 ==> 240/16 = 15
    
    #inputs = Input((img_rows, img_cols, 3))        
    inputs = Input((img_rows, img_cols, 1))
    # Normalize all data before extracting features
    input_nor = BatchNormalization()(inputs)
    
    
    # Block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(input_nor)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    
    #pool1 = Dropout(0.2)(pool1)

    # Block 2
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(pool1) 
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2 = Dropout(0.2)(pool2)
    
    # this MaxPooling2D has the same result with the above line.
    # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    # W_new = (W - F + 2P)/S + 1
    # W_new = (120 - 3 + 2*1)/2 + 1 = 60.5 --> 60 (round down)
    # P = 1, because the padding parameter is 'same', the size of W has no change
    # before divide by Stride parameter
    
   
    # Block 3
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)   
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3 = Dropout(0.2)(pool3)
    #dropout1 = Dropout(0.9)(pool3)
    
    # Block 4
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)    
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #pool4 = Dropout(0.2)(pool4)
    # dropout2 = Dropout(0.5)(pool4)
    
    # Block 5
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)    
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)   
    #conv5 = Dropout(0.2)(conv5)    
    # Block 6
    # axis = -1 same as axis = last dim order, in this case axis = 3
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), 
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(up6) 
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)    
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)    
    #conv6 = Dropout(0.65)(conv6)
    # Block 7
    up7 = concatenate([Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), 
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(up7) 
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)    
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)    
    #conv7 = Dropout(0.65)(conv7)
    # Block 8
    up8 = concatenate([Conv2DTranspose(128, kernel_size=(3,3), strides=(2, 2), 
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)    
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)    
    #conv8 = Dropout(0.65)(conv8)
    # Block 9
    up9 = concatenate([Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), 
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(up9) 
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)    
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)   
    #conv9 = Dropout(0.65)(conv9)
    
    # Block 10
    conv10 = Conv2D(nclasses, kernel_size=(1, 1), padding='same')(conv9)
    
    # Block 10
    # curr_channels = nclasses
    _, curr_width, curr_height, curr_channels = conv10._keras_shape
    out_conv10 = Reshape((curr_width * curr_height, curr_channels))(conv10)
    act_soft = Activation('softmax')(out_conv10)
    
    model = Model(inputs=[inputs], outputs=[act_soft])
    #weights=np.array([10,100,100,100,100])
    #nadam = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    sgd = SGD(lr=0.0001, decay=0.001, momentum=0.9, nesterov=True)    
    #model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 
    adam = Adam(lr=0.0001)
    #rms=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    #model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coe])
    #model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=[dice_coe])        
    model.compile(optimizer=adam,loss=bce_dice_loss,metrics=[dice_coe])
    #model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=[dice_coe])
    #model.compile(optimizer=nadam, loss=jaccard_distance_loss, metrics=[jaccard_distance])
    return model


def save_trained_agu_model(model):   

    # serialize model to JSON

    model_json = model.to_json()

    with open("cnn_BRATs_unet_new.json", "w") as json_file:    

        json_file.write(model_json)

    # serialize weights to HDF5

    model.save_weights("cnn_BRATs_unet_new.h5")    

    print("Saved model to disk")
def save_trained_model(model):
    # apply the histogram normalization method in pre-processing step    
    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_BRATs_unet_HN.json", "w") as json_file:    
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_BRATs_unet_HN.h5")   
    print("Saved model to disk")
    
def convert_data_toimage(imgs_pred):
    if imgs_pred.ndim == 3:
        nimgs, npixels, nclasses = imgs_pred.shape
        img_rows = np.sqrt(npixels).astype('int32') 
        img_cols = img_rows
        labels = True
    elif imgs_pred.ndim == 4:
        nimgs, img_rows, img_cols, _ = imgs_pred.shape
        labels = False
        
    for n in range(nimgs):
        # print(imgs_pred[n])
        if labels:
            imgs_temp = np.argmax(imgs_pred[n], axis=-1)
            # print(imgs_temp.shape)
            imgs_temp = np.reshape(imgs_temp, (img_rows, img_cols))
            # imgs_temp = imgs_temp == 4
            # print(imgs_temp.shape)            
            # c = np.count_nonzero(imgs_temp)
            # print(c)
        else:
            imgs_temp = imgs_pred[n]
            imgs_temp = np.reshape(imgs_temp, (img_rows, img_cols))
        
        imgs_temp = imgs_temp[np.newaxis, ...]
        if n == 0:
           imgs_result = imgs_temp 
        else:
           imgs_result = np.concatenate((imgs_result, imgs_temp), axis=0)           
        
    return imgs_result
    
def show_img(imgs_pred, imgs_label):
    for n in range(imgs_pred.shape[0]):
        print('Slice: %i' %n)
        img_pred = imgs_pred[n].astype('float32')
        img_label = imgs_label[n].astype('float32')               
               
        show_2Dimg(img_pred, img_label)
        
def show_2Dimg(img_1, img_2):
    fig, axes = plt.subplots(ncols=2)
    ax = axes.ravel()
    ax[0].imshow(img_1, cmap=plt.cm.gray)
    ax[1].imshow(img_2, cmap=plt.cm.gray)
    plt.show()

    
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
    imgs_train /=65535. 
    imgs_val /=65535.
    # imgs_train_1 = imgs_train[:, :, :, 0]
    # imgs_train_1 = imgs_train_1[..., np.newaxis]
    # print(imgs_train_1.shape)
    
    # define data preparation
    batch_size = 12
    datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
#==============================================================================
#     datagen = ImageDataGenerator(rotation_range=20, # rescale=1./255,
#                                  width_shift_range=0.1, height_shift_range=0.1,
#                                  horizontal_flip=True, vertical_flip=True,                                 
#                                  shear_range=0.2, fill_mode='nearest')
#==============================================================================
    # fit parameters from data
    datagen.fit(imgs_train)     
    # datagen.fit(imgs_train_1)
    train_generator = datagen.flow(imgs_train, imgs_label_train, batch_size=batch_size)
    
# =============================================================================
#     # Configure batch size and retrieve one batch of images
#     ni = 0
#     for X_batch, y_batch in datagen.flow(imgs_train_1, imgs_label_train, batch_size=batch_size):
#         ni += 1
#         # Show 9 images        
#         for i in range(0, 2):
#             print(X_batch[i].shape)
#             plt.subplot(330 + 1 + i)
#             plt.imshow(X_batch[i, :, :, 0], cmap=plt.cm.gray)
#         # show the plot
#         plt.show()
#         if ni > 4:
#             break
# =============================================================================

        
    # this is the augmentation configuration we will use for testing:
    val_datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
                                     # rescale=1./255)
    # val_datagen = ImageDataGenerator(rescale=1./255)
    
    # fit parameters from data
    val_datagen.fit(imgs_val) 
    # val_datagen.fit(imgs_train)   
    val_generator = val_datagen.flow(imgs_val, imgs_label_val, batch_size=batch_size)
    # datagen.standardize(imgs_val)
    
    print('Creating and compiling model...')    
    model = cnnBRATsInit_unet_5()
    model.summary()
    
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_dice_coe', save_best_only=True)
    
    print('Fitting model with the data augmentation ...')    
    history = model.fit_generator(train_generator,                        
                                  steps_per_epoch=(imgs_train.shape[0] // batch_size) + 1,                        
                                  epochs=80,                        
                                  verbose=1,
                                  callbacks=[model_checkpoint],
                                  validation_data=val_generator,
                                  validation_steps=(imgs_val.shape[0] // batch_size) + 1,
                    		  use_multiprocessing=True,
                                  workers=4)
    
    
    save_trained_agu_model(model)   
    
    # release memory in GPU and RAM
    del history
    del model
    for i in range(15):
        gc.collect()
        
if __name__ == '__main__':
    #train_network()
    # if the training grayscale data has channels greater than 1, it cannot 
    # be augmented using ImageDataGenerator
    train_network_aug()   
