# Evaluation of Multi-Modal MRI Images for Brain Tumor Segmentations

This is an implementation of evaluation of MRI modalities for the segementation of brain tumors with use of UNET using keras and python. 

## Installation

Clone the GitHub repository and install the dependencies.
* Install 
  * Anaconda (for creating and activating a separate environment)
  * numpy=1.13.3
  * matplotlib
  * scikit-learn==0.19.1


* Clone the repo and go to the directory 
```
$ git clone https://github.com/SafiaFatima/brats2015.git
$ cd brats2015

```

## Preprocessing
For the preprocessing of the images, applying different filters you can simply change the settings in data_prep_noaug.py

For preprocessing of the dataset :
```
python data_prep_noaug.py

```

## Training
The pretrained model is saved in models/ directory. Once you get the concept of upsampling & downsampling, you can train your own model by changing the setting in training.py.

For training the model use :
```
python training.py

```

## Testing
Testing of a trained model can be simply done by running:

```
python testing.py

```







