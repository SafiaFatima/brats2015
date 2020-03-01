

# -*- coding: utf-8 -*-
"""
@author: SAFIA FATIMA
"""
from __future__ import print_function
import os
# force tensorflow to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from skimage import data, util
from skimage import io, color
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes
import SimpleITK as sitk
from matplotlib import pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook
import h5py
import gc
import re
import csv
import matplotlib.image as mpimg
import pandas as pd

# ----------------------------------------------------------------------------
import tensorflow as tf
# import keras
from keras.models import Sequential, Model, load_model,  model_from_json
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as K
np.set_printoptions(threshold=np.inf)

# an argument to your Session's config arguments, it might help a little, 
# though it won't release memory, it just allows growth at the cost of some memory efficiency
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
# with tf.Session(config = config) as s:
sess = tf.Session(config = config)
K.set_session(sess)

# ----------------------------------------------------------------------------
from F_data_prep1 import load_test_data
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

from pandas_ml import ConfusionMatrix

# ----------------------------------------------------------------------------
# np.set_printoptions(threshold=np.inf) # help to print full array value in numpy
smooth = 1.
# img_rows = 64
# img_cols = 64
dice_0=0.0
dice_1=0.0
dice_2=0.0
dice_3=0.0
dice_4=0.0
Whole=0.0
Enhancing=0.0
Core=0.0
avgdsc=[0.0,0.0,0.0,0.0,0.0]
avgppv=[0.0,0.0,0.0,0.0,0.0]
avgsen=[0.0,0.0,0.0,0.0,0.0]
#DSC=[]
#PPV=[]
#Sensitivity=[]
def classification_report_to_csv_pandas_way(ground_truth,
                                            predictions,
                                            full_path="test_pandas.csv"):
    """
    Saves the classification report to csv using the pandas module.
    :param ground_truth: list: the true labels
    :param predictions: list: the predicted labels
    :param full_path: string: the path to the file.csv where results will be saved
    :return: None
    """
    import pandas as pd

    # get unique labels / classes
    # - assuming all labels are in the sample at least once
    labels = unique_labels(ground_truth, predictions)

    # get results
    precision, recall, f_score, support = precision_recall_fscore_support(ground_truth,
                                                                          predictions,
                                                                          labels=labels,
                                                                          average=None)
    # a pandas way:
    results_pd = pd.DataFrame({"class": labels,
                               "precision": precision,
                               "recall": recall,
                               "f_score": f_score,
                               "support": support
                               })
    #print (results_pd['class'].values[1])
    #results_pd.to_csv(full_path, index=False)
    return results_pd

def dice_coef(y_true, y_pred):
    #y_true_f = K.flatten(y_true.astype('float32'))
    #y_pred_f = K.flatten(y_pred.astype('float32'))
    y_true_f = K.flatten(y_true) # K.flatten(y_true.astype('float32'))
    y_pred_f = K.flatten(y_pred) # K.flatten(y_pred.astype('float32'))
    print (y_true_f)
    print ("******************")
    print (y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)



def dice_score(y_pred, y_true):
    y_true_f = y_true.flatten() # K.flatten(y_true.astype('float32'))
    y_pred_f = y_pred.flatten() # K.flatten(y_pred.astype('float32'))
    intersection = np.count_nonzero(y_true_f * y_pred_f)
    
    return (2. * intersection) / (np.count_nonzero(y_true_f) + np.count_nonzero(y_pred_f))

def dice_score_full(y_pred, y_true):
    # dice coef of entire tumor
    y_true_f = y_true.flatten() 
    y_pred_f = y_pred.flatten()
    print (classification_report(y_true_f,y_pred_f)) 
    conf_matrix=confusion_matrix(y_true_f,y_pred_f)
    print (confusion_matrix(y_true_f,y_pred_f))
     

    
    


    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    #TN = conf_matrix.values.sum() - (FP + FN + TP)
    DSC=[None]*5
    PPV=[None]*5
    Sensitivity=[None]*5

    print ("TP:",TP)
    print ("FP:",FP)
    print ("FN:",FN)

    

    PredictedClasses=classification_report_to_csv_pandas_way(y_true_f,y_pred_f)
    
    

    counterForTPFNFP = 0
    for i in range(len(PredictedClasses['class'].values)):
                
        classNumber=int(PredictedClasses['class'].values[i])

        #print (classNumber)

        DSC[classNumber]=2*TP[counterForTPFNFP]/(FP[counterForTPFNFP]+(2*TP[counterForTPFNFP])+FN[counterForTPFNFP])
        PPV[classNumber]=TP[counterForTPFNFP]/(TP[counterForTPFNFP]+FP[counterForTPFNFP])
        Sensitivity[classNumber]=TP[counterForTPFNFP]/(TP[counterForTPFNFP]+FN[counterForTPFNFP])

        if (np.isnan(PPV[classNumber])):
            #print ("True")
            PPV[classNumber]=0.0
        if (np.isnan(Sensitivity[classNumber])):
            #print ("True")
            Sensitivity[classNumber]=0.0 

        counterForTPFNFP+=1
        print ("---")
        
    global dice_0,dice_1,dice_2,dice_3,dice_4
    nclasses = 5
    dice=np.zeros(nclasses)
    denominator=np.zeros([nclasses])
    
    for i in range(len(PredictedClasses['class'].values)):

        classNumber=int(PredictedClasses['class'].values[i])
        y_true_i=np.equal(y_true_f,classNumber)
        y_pred_i=np.equal(y_pred_f,classNumber)
        denominator[classNumber]=np.sum(y_true_i)+np.sum(y_pred_i)
        dice[classNumber]=(2. * np.sum(y_true_i*y_pred_i))/denominator[classNumber]

        print ("dice of ",classNumber,"is :",dice[classNumber])
   
    dice_0+=dice[0]
    dice_1+=dice[1]
    dice_2+=dice[2]
    dice_3+=dice[3]
    dice_4+=dice[4]
        
    intersection = np.count_nonzero(y_true_f * y_pred_f)

    print ("Intersection is :", intersection)
    if intersection > 0:
        whole_tumor = (2. * intersection) / (np.count_nonzero(y_true_f) + np.count_nonzero(y_pred_f))
    else:
        whole_tumor = 0
    
    # dice coef of enhancing tumor
    enhan_gt = np.argwhere(y_true == 4)
    gt_a, seg_a = [], [] # classification of
    for i in enhan_gt:
        gt_a.append(y_true[i[0]][i[1]])
        seg_a.append(y_pred[i[0]][i[1]])
    gta = np.array(gt_a)
    sega = np.array(seg_a)
    if len(enhan_gt) > 0:
        enhan_tumor = float(len(np.argwhere(gta == sega))) / float(len(enhan_gt))
    else:
        enhan_tumor = 0
    
    # dice coef core tumor
    noenhan_gt = np.argwhere(y_true == 3)
    necrosis_gt = np.argwhere(y_true == 1)
    live_tumor_gt = np.append(enhan_gt, noenhan_gt, axis = 0)
    core_gt = np.append(live_tumor_gt, necrosis_gt, axis = 0)
    gt_core, seg_core = [], []
    for i in core_gt:
        gt_core.append(y_true[i[0]][i[1]])
        seg_core.append(y_pred[i[0]][i[1]])
    gtcore, segcore = np.array(gt_core), np.array(seg_core)
    if len(core_gt) > 0:
        core_tumor = float(len(np.argwhere(gtcore == segcore))) / float(len(core_gt))
    else:
        core_tumor = 0
    
    return whole_tumor, enhan_tumor, core_tumor

def load_trained_agu_model():   
    # load json and create model    
    json_file = open('NewUnetBceNormFlair/cnn_BRATs_unet_new.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model    
    loaded_model.load_weights("NewUnetBceNormFlair/cnn_BRATs_unet_new.h5")
    print("Loaded model from disk")
    
    return loaded_model
 
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
            print(imgs_temp.shape)
            imgs_temp = np.reshape(imgs_temp, (img_rows, img_cols))
            # imgs_temp = imgs_temp == 4
            # print(imgs_temp.shape)            
            c = np.count_nonzero(imgs_temp)
            print(c)
        else:
            imgs_temp = imgs_pred[n]
            imgs_temp = np.reshape(imgs_temp, (img_rows, img_cols))
        
        imgs_temp = imgs_temp[np.newaxis, ...]
        if n == 0:
           imgs_result = imgs_temp 
        else:
           imgs_result = np.concatenate((imgs_result, imgs_temp), axis=0)           
        
    return imgs_result.astype('int16')

def show_img(imgs_pred, imgs_label):
    F=open('testfile.txt','w')
    global Whole, Enhancing, Core

    #count=0
    for n in range(imgs_pred.shape[0]):
        print('Slice: %i' %n)
        img_pred = imgs_pred[n].astype('float32')
        img_label = imgs_label[n].astype('float32')
        
        #dice = dice_score(imgs_pred[n], imgs_label[n])
        #print("Dice score: %.3f" %dice)
        whole_tumor, enhan_tumor, core_tumor = dice_score_full(img_pred, img_label)
        print("Whole tumor: %.3f, Enhancing tumor: %.3f, Core: %.3f" % (whole_tumor, enhan_tumor, core_tumor))
        
        Whole+=whole_tumor
        Enhancing+=enhan_tumor
        Core+=core_tumor



        F.write(str(n))
        F.write (" , ")
        F.write (str(whole_tumor))
        F.write (" , ")
        F.write (str(enhan_tumor))
        F.write(" , ")
        F.write (str(core_tumor))
        F.write("\n")
        #show_2Dimg(imgs_pred[n], imgs_label[n])
        
        
    F.close()
                
def show_2Dimg(img_1, img_2):
    fig, axes = plt.subplots(ncols=2)
    ax = axes.ravel()
    img_11 = cvt2color_img(img_1)
    img_22 = cvt2color_img(img_2)
    ax[0].imshow(img_11,cmap=plt.cm.gray)
    ax[1].imshow(img_22,cmap=plt.cm.gray)
    plt.show()

def show_img_32(imgs_seg1, imgs_seg2):#, imgs_seg3):

    patient_HGG1_1=[28]
    first_slice_in_each_patient_HGG1_1=[48]
    patient_HGG1_2=[57]
    first_slice_in_each_patient_HGG1_2=[41]
    patient_HGG2_2=[33]
    first_slice_in_each_patient_HGG2_2=[80]
    patient_HGG3_1=[38]
    first_slice_in_each_patient_HGG3_1=[75]
    patient_HGG3_2=[32]
    first_slice_in_each_patient_HGG3_2=[24]
    patient_HGG4_1=[25]
    first_slice_in_each_patient_HGG4_1=[53]
    patient_HGG4_2=[22]
    first_slice_in_each_patient_HGG4_2=[99]
    patient_HGG5_1=[33]
    first_slice_in_each_patient_HGG5_1=[102]
    patient_HGG6_1=[61]
    first_slice_in_each_patient_HGG6_1=[47]
    patient_HGG6_2=[33]
    first_slice_in_each_patient_HGG6_2=[76]
    patient_HGG8_2=[42]
    first_slice_in_each_patient_HGG8_2=[72]
    patient_HGG9_1=[25]
    first_slice_in_each_patient_HGG9_1=[37]
    patient_HGG10_1=[3]
    first_slice_in_each_patient_HGG10_1=[58]
    patient_HGG10_2=[22]
    first_slice_in_each_patient_HGG10_2=[52]
    patient_HGG11_1=[25]
    first_slice_in_each_patient_HGG11_1=[51]
    patient_HGG11_2=[13]
    first_slice_in_each_patient_HGG11_2=[47]
    patient_HGG11_4 = [29]
    first_slice_in_each_patient_HGG11_4 = [90]
    patient_HGG11_4 = [21]
    first_slice_in_each_patient_HGG11_4 = [51]
    patient_HGG1_3 = [31]
    first_slice_in_each_patient_HGG1_3 = [75]
    patient_HGG1_4 = [40]
    first_slice_in_each_patient_HGG1_4 = [46]
    patient_HGG2_2 = [21]
    first_slice_in_each_patient_HGG2_2 = [44]
    patient_HGG12_2 = [37]
    first_slice_in_each_patient_HGG12_2 = [67]
    patient_HGG11_3 = [22]
    first_slice_in_each_patient_HGG11_3 = [90]

    
    



    patients = patient_HGG11_3
    first_slice_in_each_patient = first_slice_in_each_patient_HGG11_3
    slice_number = 0
    counter = 0
    index = 0
    patient_number = 1
    slice_number = first_slice_in_each_patient[index]
    for n in range(imgs_seg1.shape[0]):
        print('Slice: %i' %n)
        print (patient_number)
        print (slice_number)
        counter = n
        
        if (counter == patients[index]):
            index+=1
            patient_number+=1
            slice_number = first_slice_in_each_patient[index]

        img_seg1 = imgs_seg1[n].astype('float32')
        img_seg2 = imgs_seg2[n].astype('float32')
        
        #img_label = imgs_seg3[n].astype('float32')
                
        img_1 = cvt2color_img(img_seg1,patient_number,slice_number)
        img_2 = cvt2color_img(img_seg2,patient_number,slice_number) 
        
        show_2Dimg_3(img_1, img_2,patient_number,slice_number)#, img_3)
        slice_number+=1       
        #img_3 = cvt2color_img(img_label)
 
def show_2Dimg_3(img_1, img_2, patient_number, slice_number):
    #fig, axes = plt.subplots(ncols=3)

    fig, axes = plt.subplots(ncols=2)
    ax = axes.ravel()
    ax[0].imshow(img_2, cmap=plt.cm.gray)
    ax[0].set_xlabel('Ground Truth',fontsize = 14.0)

    ax[0].axes.get_xaxis().set_ticks([])
    ax[1].axes.get_xaxis().set_ticks([])
    ax[0].axes.get_yaxis().set_ticks([])
    ax[1].axes.get_yaxis().set_ticks([])

    ax[1].imshow(img_1, cmap=plt.cm.gray)
    ax[1].set_xlabel('Prediction',fontsize = 14.0)

    fig.savefig('/home/res-safia-fatima/Flair/NewUnetBceNormFlair/HGG11_3/'+'Patient'+str(patient_number)+'Slice'+str(slice_number)+'.png')   
# save the figure to file
    plt.close(fig) 

def cvt2color_img(img_src,patient_number,slice_number):
    imgddd = mpimg.imread('/home/res-safia-fatima/Flair/TestingFlair/HGG11_3/threshold_patient_HGG11_3/'+'Patient'+str(patient_number)+'Slice'+str(slice_number)+'.png')
    ones = np.argwhere(img_src == 1) # class 1/necrosis
    twos = np.argwhere(img_src == 2) # class 2/edema
    threes = np.argwhere(img_src == 3) # class 3/non-enhancing tumor
    fours = np.argwhere(img_src == 4) # class 4/enhancing tumor
 
    #print (pix)
    img_dst = color.gray2rgb(imgddd)
    red_multiplier = [1, 0.2, 0.2] # class 1/necrosis    
    green_multiplier = [0.35, 0.75, 0.25] # class 2/edema
    blue_multiplier = [0, 0.25, 0.9] # class 3/non-enhancing tumor
    yellow_multiplier = [1, 1, 0.25] # class 4/enhancing tumor
# =============================================================================
#     img_dst[img_src == 1] = red_multiplier
#     img_dst[img_src == 2] = green_multiplier
#     img_dst[img_src == 3] = blue_multiplier
#     img_dst[img_src == 4] = yellow_multiplier
# =============================================================================
 
    # change colors of segmented classes
    #print ("*********** Ones :", ones)
    for i in range(len(ones)):
        img_dst[ones[i][0]][ones[i][1]] = red_multiplier        
    for i in range(len(twos)):
        img_dst[twos[i][0]][twos[i][1]] = green_multiplier        
    for i in range(len(threes)):
        img_dst[threes[i][0]][threes[i][1]] = blue_multiplier
    for i in range(len(fours)):
        img_dst[fours[i][0]][fours[i][1]] = yellow_multiplier
    
    #print (img_src)

    return img_dst
        
def save_result_h5py(nda):
    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('data_1', data=nda)
    h5f.close()
    print("Saving data to disk done.")


def test_network_aug():
    batch_size = 12   
    
    print('Loading and preprocessing test data...')
    imgs_test, imgs_label_test = load_test_data()
    print('Imgs test shape', imgs_test.shape)
    print('Imgs test label shape', imgs_label_test.shape)
                
    # imgs_test = preprocessing(imgs_test)
    imgs_test = imgs_test.astype('float32') 
    imgs_test /= 65535.
    imgs_test_ref = imgs_test

    test_datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)    

    # fit parameters from data
    test_datagen.fit(imgs_test)     
    test_generator = test_datagen.flow(imgs_test, imgs_label_test, batch_size=batch_size)  
    
    print('Loading the trained model...')
    model = load_trained_agu_model()
    model.summary()
    
    print('Predicting model with test data...')
    prop = model.predict_generator(test_generator, steps=(imgs_test.shape[0] // batch_size) + 1,verbose=1)
    print(prop.shape)
    print ("Converting to image")    
    imgs_test_pred = convert_data_toimage(prop)
    print(imgs_test_pred.shape)
    # maxv = np.max(imgs_test_pred)
    # print(maxv)
    imgs_label_test = convert_data_toimage(imgs_label_test)
    print(imgs_label_test.shape)
    show_img(imgs_test_pred,imgs_label_test)
    dsc1,ppv1,sen1=0.0,0.0,0.0
    for i in range(len(avgdsc)):

       print ("class",i)
       dsc1+=avgdsc[i]
       ppv1+=avgppv[i]
       sen1+=avgsen[i]
       print ("avgdsc:",avgdsc[i]/imgs_test_pred.shape[0])
       print ("avgppv:",avgppv[i]/imgs_test_pred.shape[0])
       print ("avgsen:",avgsen[i]/imgs_test_pred.shape[0])

    print ("\n")
    print ("Whole Tumor :", Whole/imgs_test_pred.shape[0])
    print ("Enhancing Tumor :", Enhancing/imgs_test_pred.shape[0])
    print ("Core Tumor :", Core/imgs_test_pred.shape[0])
    print ("Dice 0 :",dice_0/imgs_test_pred.shape[0])
    print ("Dice 1 :",dice_1/imgs_test_pred.shape[0]) 
    print ("Dice 2 :",dice_2/imgs_test_pred.shape[0])
    print ("Dice 3 :",dice_3/imgs_test_pred.shape[0])   
    print ("Dice 4 :",dice_4/imgs_test_pred.shape[0])
 

    YT=[]
    YP=[]
    for n in range(imgs_test_pred.shape[0]):


        y_true = imgs_label_test[n].flatten() 
        y_pred = imgs_test_pred[n].flatten() 

        YT = np.append( YT , y_true)
        YP = np.append( YP ,  y_pred)



    print ('\n')
    print ('-------------------------------------------')
    print ('\n')

    #print (confusion_matrix(YT,YP,classes=[0,1,2,3,4]))
    print (ConfusionMatrix(YT, YP))
    conf_matrix=confusion_matrix(YT,YP)

    


    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)

    print ('\n')
    print ("TP:",TP)
    print ("FP:",FP)
    print ("FN:",FN)    
        

    

    print ('\n')
    print ('-------------------------------------------')
    print ('\n')
    
    #show_img_32(imgs_pred_post, imgs_test_pred, imgs_label_test)
    show_img_32(imgs_test_pred,imgs_label_test)
        
if __name__ == '__main__':

    test_network_aug()
    

