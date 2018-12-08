# Python

import sys
import os
import datetime
import glob
from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import cv2
from skimage.io import *
from skimage.color import * 
import skimage.transform                              
from skimage.morphology import label  

# keras
import keras
from keras.models import *
from keras.layers import Input, merge, BatchNormalization, AtrousConvolution2D, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.merge import add, concatenate

from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import  ModelCheckpoint, CSVLogger, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping

from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split


from keras.utils import np_utils
from keras.datasets import mnist

# XGBOOST

import sys
import os
import datetime
import glob
from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import cv2
from skimage.io import *
from skimage.color import * 
import skimage.transform                              
from skimage.morphology import label  

import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from subprocess import check_output
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score