# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:43:24 2017

@author: Gavrilov
"""

import IPython
from IPython.display import display, Audio

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt

from PIL import Image #Imaging Library #https://pillow.readthedocs.io/en/4.3.x/

import tensorflow as tf
from tensorflow.python.framework import ops

import keras
import keras.backend as K #probably this is for GPU, so I don't need this
from keras.preprocessing.text import Tokenizer

from keras_tqdm import TQDMNotebookCallback #https://github.com/bstriner/keras-tqdm
from keras import initializers
from keras.applications.resnet50 import ResNet50, decode_predictions, conv_block, identity_block
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors, LSHForest

import numpy as np
from numpy.random import normal

import pandas as pd

import scipy

from gensim.models import word2vec #https://anaconda.org/anaconda/gensim

from nltk.tokenize import ToktokTokenizer, StanfordTokenizer

import threading #Thread-based parallelism #https://docs.python.org/3/library/threading.html
import shutil #High-level file operations #https://docs.python.org/3/library/shutil.html
import functools #Higher-order functions and operations on callable objects #https://docs.python.org/3/library/functools.html
from functools import reduce
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor #Launching parallel tasks #https://docs.python.org/3/library/concurrent.futures.html

import re #Regular expression operations #https://docs.python.org/3/library/re.html
import glob #Unix style pathname pattern expansion #https://docs.python.org/3/library/glob.html

import math #https://docs.python.org/3/library/math.html
import datetime #https://docs.python.org/3/library/datetime.html
import collections #Container datatypes #https://docs.python.org/3/library/collections.html

import json #JSON encoder and decoder #https://docs.python.org/3/library/json.html #probably needed for GPU, so I don't need this

import operator #Standard operators as functions #https://docs.python.org/3/library/operator.html
import itertools #Functions creating iterators for efficient looping #https://docs.python.org/3/library/itertools.html
from itertools import chain
import random

import tarfile #Read and write tar archive files #https://docs.python.org/3/library/tarfile.html
import pickle 
import os #Miscellaneous operating system interfaces #https://docs.python.org/3/library/os.html

import xgboost #implementation of gradient boosted decision trees #http://xgboost.readthedocs.io/en/latest/python/python_intro.html#
import bcolz