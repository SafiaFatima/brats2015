import numpy as np

#imgs_train = np.load('mhafiles/DataLearnIntensityTrue/imgs_label_train_unet_HG.npy')
#imgs_train2 = np.load('mhafiles/DataLearnIntensityTrue2/imgs_label_train_unet_HG2.npy')
#imgs_train3 = np.load('mhafiles/DataLearnIntensityTrue3/imgs_label_train_unet_HG3.npy')
#imgs_train4 = np.load('mhafiles/DataLearnIntensityTrue4/imgs_label_train_unet_HG4.npy')
#imgs_train5 = np.load('mhafiles/DataLearnIntensityTrue5/imgs_label_train_unet_HG5.npy')
#imgs_train6 = np.load('mhafiles/DataLearnIntensityTrue6/imgs_label_train_unet_HG6.npy')
#imgs_train7 = np.load('mhafiles/DataLearnIntensityTrue7/imgs_label_train_unet_HG7.npy')
imgs_train8 = np.load('mhafiles/DataLearnIntensityTrue8/imgs_label_train_unet_HG8.npy')
imgs_train9 = np.load('mhafiles/DataLearnIntensityTrue9/imgs_label_train_unet_HG9.npy')
imgs_train10 = np.load('mhafiles/DataLearnIntensityTrue10/imgs_label_train_unet_HG10.npy')
imgs_train11 = np.load('mhafiles/DataLearnIntensityTrue11/imgs_label_train_unet_HG11.npy')

eigth=np.load('mhafiles/DataLearnIntensityTrueFinal/imgs_label_train_unet_eigth.npy')





#print('Imgs train shape', imgs_train.shape)
#print('Imgs train2 shape', imgs_train2.shape)
#print('Imgs train3 shape', imgs_train3.shape)
#print('Imgs train4 shape', imgs_train4.shape)
#print('Imgs train5 shape', imgs_train5.shape)
#print('Imgs train6 shape', imgs_train6.shape)
#print('Imgs train7 shape', imgs_train7.shape)
print('Imgs train8 shape', imgs_train8.shape)
print('Imgs train9 shape', imgs_train9.shape)
print('Imgs train10 shape', imgs_train10.shape)
print('Imgs train11 shape', imgs_train11.shape)


#eight=np.concatenate((imgs_train,imgs_train2,imgs_train3,imgs_train4,imgs_train5,imgs_train6,imgs_train7),axis=0)
#print (eight.shape)
#np.save('mhafiles/DataLearnIntensityTrueFinal/imgs_label_train_unet_eigth.npy', eight)

restOfFive=np.concatenate((imgs_train8,imgs_train9,imgs_train10,imgs_train11,eigth),axis=0)
print (restOfFive.shape)
np.save('mhafiles/DataLearnIntensityTrueFinal/imgs_label_train_unet_HGrestOfFive.npy', restOfFive)
