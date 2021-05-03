import os
import argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "" #then these two lines force keras to use your CPU
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, \
LeakyReLU, Conv2DTranspose, ReLU, Reshape, Concatenate, Input
from keras.utils import to_categorical
from tensorflow.image import psnr
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

from EMDataGenerator import EMDataGenerator
  # not scipy sparse because that is not how michael encoded it

def get_args():
	parser = argparse.ArgumentParser(allow_abbrev=False)

	# Add data arguments
	parser.add_argument("--data_path", default="../em_data/em_data_32x32_043021/", help="path to data directory")
	parser.add_argument("--out_path", default="'../models/model_050321/'", help="train dataset name")
	parser.add_argument("--grid_size", default=32, type=int, help="voxel and image dimensions")
	parser.add_argument("--num_epochs", default=100, type=int, help="force stop training at specified epoch")

	args = parser.parse_args()
	print("vars(args)",vars())
	return args

## Inputs
args = get_args()
grid_size = args.grid_size
X_filename = args.data_path+ 'X_list_{}x{}x{}.pkl'.format(grid_size,grid_size,grid_size)
y_filename = args.data_path+ 'y_list_full_{}x{}.pkl'.format(grid_size,grid_size)
defocus_filename = args.data_path+ 'defocus_list_{}x{}.pkl'.format(grid_size,grid_size)

num_epochs = args.num_epochs
sample_shape = (grid_size,grid_size,grid_size,2)

# This is a bit confusing, but train and validation are randomly selected from train_val_range so that 
# validation is just new defocus views of lattices we've seen
# Test is held out lattices, all of test_range.
train_val_range = [0.,0.9]
train_shuffle = [0.,0.9]
val_shuffle = [0.9,1.]
test_range = [0.9,1.]
train_generator = EMDataGenerator(X_filename,y_filename,train_val_range,train_shuffle,5,defocus_filename)
valid_generator = EMDataGenerator(X_filename,y_filename,train_val_range,val_shuffle,5,defocus_filename)
test_generator = EMDataGenerator(X_filename,y_filename,test_range,None,5,defocus_filename)

## Training Metrics
# Calculate training y pixel intensity range, for PSNR function
y_range = np.array(train_generator.y).max() - np.array(train_generator.y).min()


# def psnr_metric_orig(y_true, y_pred):
#     mse = keras.losses.MSE(y_true,y_pred).numpy()
#     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#                   # Therefore PSNR have no importance.
#         return 100
#     psnr = 20 * np.log10(y_range / np.sqrt(mse))
#     return psnr

def psnr_metric(y_true,y_pred):
    return psnr(y_true,y_pred,y_range)

def mse_metric(y_true,y_pred):
    return keras.losses.MSE(y_true,y_pred)


## Model setup

def conditional_model(sample_shape=(32,32,32,2), defocus_1hot_shape = (9,)):
  '''
  takes in 3D input of shape (dim,dim,dim,2) and outputs a grey-scale 2-D image of shape (dim,dim,1).
  dim should be an power of 2 integer
  sample_shape:
    the shape of 1 sample, should be (dim,dim,dim,2)
  defocus_1hot_shape_shape:
    the shape of 1 row of one-hot-endoed defocus parameter of the input sample, should be (#_unique_defocus-1, )
  '''

  dim = sample_shape[0]
  f = int(32/(dim/32))
    # the starting filter size

  iter = int(np.log2(dim)) - 1
    # takes one less iteration because we want the shape to stop at 2, not 1
    # 4 for 32, 5 for 64, 6 for 128, 7 for 256

  # Create the model
  input_voxel = Input(shape=sample_shape)
  defocus = Input(shape=defocus_1hot_shape, dtype='float32')
  
  # add first layer that takes in the input
  encoder = Conv3D(filters=f, 
                  kernel_size=(4, 4, 4), 
                  strides=(2, 2, 2),
                  padding='same', 
                      # `same` just means as long as even just the left most 1 column of your kernel is still in the sample matrix, you will use padding to fill the parts that ran over the matrix and finish that mapping
                      # if you keep moving till you kernel does not overlap with your matrix at all we will stop and won't pad anymore
                  use_bias=False,
                  input_shape=sample_shape)(input_voxel)
  # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
  encoder = BatchNormalization(center=True, scale=True)(encoder)
  encoder = LeakyReLU(alpha=0.2)(encoder)
  # now the output should have shape (dim/2, dim/2, dim/2, f)

  # add middel encoder layers
  for i in range(1, iter): # 1~iter-1
    encoder = Conv3D(filters=f*(2**i), 
                    kernel_size=(4, 4, 4), 
                    strides=(2, 2, 2),
                    padding='same',
                    use_bias=False)(encoder)
    encoder = BatchNormalization(center=True, scale=True)(encoder)
    encoder = LeakyReLU(alpha=0.2)(encoder)
  
  # now the output should have shape (2, 2, 2, f*(2**i))

  # add latent layer
  encoder = Conv3D(filters=100, 
                  kernel_size=(2, 2, 2), 
                  strides=(1, 1, 1),
                  padding='valid',
                  use_bias=False)(encoder)
  encoder = LeakyReLU(alpha=0.2)(encoder)
    # VALID : Don't apply any padding
    # now the output should have shape (1, 1, 1, 100)

  encoder = Reshape((1,1,100))(encoder)
    # must reshape from a 3-D structure with 100 channels to a 2-D image having 100 channels
    # so Conv2DTranspose can work properly
    # now the output should have shape (1, 1, 100)

  defocus_vector = Reshape((1,1,defocus_1hot_shape[0]))(defocus)
    # reshape defocus vector to have shape (1, 1, defocus_input_shape)
    # must have the same dimension shape to concatenate with the latent vector
  latent_vector = Concatenate()([encoder, defocus_vector])
    # now the output should have shape (1,1,100+defocus_1hot_shape) now

  # add first blow-up decoder layer
  decoder = Conv2DTranspose(filters=f*(2**i),
                            kernel_size=(2,2),
                            strides=(1,1),
                            padding='valid',
                            use_bias=False
                            )(latent_vector)
  decoder = BatchNormalization(center=True, scale=True)(decoder)
  decoder = ReLU()(decoder)
  # now the output should have shape (2, 2, f*(2**i))

  # add middle decoder layers
  for i in range(iter-2, -1, -1 ): #iter-2 ~ 0
    decoder = Conv2DTranspose(filters=f*(2**i),
                              kernel_size=(4,4),
                              strides=(2,2),
                              padding='same',
                              use_bias=False
                              )(decoder)
    decoder = BatchNormalization(center=True, scale=True)(decoder)
    decoder = ReLU()(decoder)
  
  # now the output should have shape (dim/2, dim/2, f)

  # add final deocder output layer
  decoder = Conv2DTranspose(filters=1,
                            kernel_size=(4,4),
                            strides=(2,2),
                            padding='same',
                            use_bias=False
                            )(decoder)
  img = Dense(1, activation='tanh')(decoder)
  # now the output should have shape (dim, dim, 1)

  cond_model = Model([input_voxel,defocus], img)

  return cond_model

# Compile the model
cond_model = conditional_model(sample_shape)
cond_model.compile(loss='mean_squared_error',
                   optimizer=keras.optimizers.Adam(lr=0.001),
                   metrics=[psnr_metric,mse_metric])
print("input shape:", sample_shape)
cond_model.summary()


# Train the model
# Fit with DataGenerators
history = cond_model.fit(train_generator,
          validation_data = valid_generator,
          epochs = num_epochs)

## Save the model and training history
# Create model directory
model_dir = args.out_path
model_file = 'model_{}x{}_{}epochs_050321.pth'.format(grid_size,grid_size,num_epochs)

# Save model
cond_model.save(model_dir+model_file)

# Save history file

hist_file = "history.pkl"
with open(model_dir+'history.pkl', 'wb') as f:
    pickle.dump(history.history, f)



