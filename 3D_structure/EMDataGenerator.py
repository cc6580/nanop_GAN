import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose, ReLU, Reshape
# from keras.models import Sequential
# from keras.layers import Dense, 
from tensorflow.keras.utils import to_categorical
import torch 
from sklearn.preprocessing import OneHotEncoder
import random
import itertools
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sparse


class EMDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_filename, y_filename, use_range, shuffle_range, batch_size, defocus_filename=None):
        self.X_filename = X_filename
        self.y_filename = y_filename
        self.defocus_filename = defocus_filename
        self.use_range = use_range
        self.shuffle_range = shuffle_range
        
        self.batch_size = batch_size
        self.X = self.load_pickle(X_filename) # List of arrays
        self.y = self.load_pickle(y_filename) # List of arrays
        self.oe = self.train_one_hot()
        if defocus_filename:
            self.defocus_params = self.one_hot_defocus(self.load_pickle(defocus_filename))

    def one_hot_defocus(self,defocus):
        return self.oe.transform(np.array(defocus).reshape(-1,1))

    def train_one_hot(self):
        oe = OneHotEncoder(drop='first', sparse=False) 
        oe.fit(np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1))
        return oe

    def load_pickle(self,filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        first_idx = int(len(data)*self.use_range[0])
        last_idx =  int(len(data)*self.use_range[1])
        data_out = data[first_idx:last_idx]
        
        if self.shuffle_range:
            random.seed(5)
            random.shuffle(data_out)
            first_idx_shuffle = int(len(data_out)*self.shuffle_range[0])
            last_idx_shuffle = int(len(data_out)*self.shuffle_range[1])
            return data_out[first_idx_shuffle:last_idx_shuffle] 
        else:
            return data_out

    def __len__(self):
        'Denotes the number of batches per epoch'
        sample_count = len(self.X)
        return int(sample_count / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        X = np.asarray([sparse.COO.todense(xi) for xi in self.X[index*self.batch_size:(index+1)*self.batch_size]])
        y = np.asarray(self.y[index*self.batch_size:(index+1)*self.batch_size])
        y = np.expand_dims(y,axis=-1)
        # Adjust y to [0,1]
        y_train = y[:int(y.shape[0]*0.8)]
        y_adj = (y-y_train.min()) / (y_train.max()-y_train.min())
        if self.defocus_filename:
            defocus_params = self.defocus_params[index*self.batch_size:(index+1)*self.batch_size,:]
            return [X, defocus_params], y_adj
        else:
            return X, y_adj