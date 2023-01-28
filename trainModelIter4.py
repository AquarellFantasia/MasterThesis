######################################################################
############################# Imports ################################
######################################################################
import sys
import os
import uuid

import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import Sequence
from keras.utils import load_img
from keras.callbacks import *

import keras.backend as K

###################################################################### 
#################### Get arguments for script ########################
######################################################################
# sys.argv[1] - Epochs
# sys.argv[2] - optimizer
# sys.argv[3] - loss function name
# sys.argv[4] - image data path csv
# sys.argv[5] - verbose. 0 - no epoch logs; 1 - all logs; 2 - only epochs 
# sys.argv[6] - name to differentiate the files from one another
# sys.argv[7] - folder for output files
# sys.argv[8] - model function name

# load defaults
epochs      = int(sys.argv[1])
optimizer   = eval(sys.argv[2])
loss_string = sys.argv[3]
csv_path    = sys.argv[4]
verbose     = int(sys.argv[5])
unique_name = sys.argv[6]
out_path    = sys.argv[7]
model_name  = sys.argv[8]



print("Epochs: ", sys.argv[1])
print("Optimizer: ", sys.argv[2])
print("Loss function name: ", sys.argv[3])
print("Csv file used: ", sys.argv[4])
print("Verbose: ", sys.argv[5])
print("Unique name: ", sys.argv[6])
print("Output folder: ", sys.argv[7])
print("Model name: ", sys.argv[8])

input_size = 500

# image id, to easily find it
random_id = uuid.uuid1()
random_id_str = random_id.hex

######################################################################    
####################### Load refference data #########################    
######################################################################       
   
df = pd.read_csv(csv_path)
train_df = df[df['ImagePath'].str.contains("train")]
test_df = df[df['ImagePath'].str.contains("test")]
valid_df = df[df['ImagePath'].str.contains("valid")]

######################################################################    
########################## data generator ############################    
######################################################################    
class datagenerator(tf.keras.utils.Sequence):
    def __init__(self, 
            batch_size, 
            img_size,
            data_paths_df,
            input_channels,
            output_channels):
         
        self.batch_size = batch_size
        self.img_size = img_size
        self.data_paths_df = data_paths_df
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.data_paths = data_paths_df.values[:,1]
        self.params = data_paths_df.values[:,3:6]
        assert len(self.data_paths) == len(self.params)
        
        self.n = len(self.data_paths)

    def on_epoch_end(self):
        'updates indexes after each epoch'
        self.data_paths_df = self.data_paths_df.sample(frac = 1)
        self.data_paths = self.data_paths_df.values[:,1]
        self.params = self.data_paths_df.values[:,3:6]
    
    def __getitem__(self, index):
        batch_data_paths = self.data_paths[index : index + self.batch_size]
        batch_params_paths = self.params[index : index + self.batch_size]

        return self.__dataloader(self.img_size,
                batch_data_paths, batch_params_paths,
                self.input_channels, self.output_channels)
    
    def __len__(self):
        return self.n // self.batch_size

    def __dataloader(self, 
            img_size,
            data_paths,
            batch_params_paths,
            input_channels,
            output_channels):
        x = np.zeros((len(data_paths), img_size[0], img_size[1], input_channels))
        y = batch_params_paths
        
        
        for i in range(len(data_paths)):
            data = load_img(path = data_paths[i], color_mode = "grayscale")
            data = tf.keras.utils.img_to_array(data, data_format="channels_last", dtype="float32")
            data /= 255
            data.shape = (1,) + data.shape
            x[i] = np.asarray(data)
        return x.astype("float32"), np.array(y).astype("float32")
    
######################################################################    
######################## Loading the model ###########################    
###################################################################### 
def load_model_a():
    print(''' 
        ################ MODEL ############### \n 
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(3)(x)

    ''')

    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
    
def load_model_b():
    print(''' 
        ################ MODEL ############### \n 
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=16, kernel_size=7, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(3)(x)
    ''')

    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=7, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_c():
    print(''' 
        ################ MODEL ############### \n 
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=16, kernel_size=7, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x) 
        outputs = layers.Dense(3)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
    ''')
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=7, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x) 
    outputs = layers.Dense(3)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_d():
    print(''' 
        ################ MODEL ############### \n 
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=16, kernel_size=7, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x) 
        outputs = layers.Dense(3)(x)
    ''')
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=7, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x) 
    outputs = layers.Dense(3)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_e():
    print(''' 
        ################ MODEL ############### \n 
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=4, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=4, kernel_size=7, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=4, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l1(0.01))(x) 
        outputs = layers.Dense(3)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
    ''')
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=4, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=4, kernel_size=7, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=4, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l1(0.01))(x) 
    outputs = layers.Dense(3)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_f():
    print(''' 
        ################ MODEL ############### \n
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=4, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=4, kernel_size=7, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l1(0.01))(x) 
        outputs = layers.Dense(3)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
    ''')
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=4, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=4, kernel_size=7, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l1(0.01))(x) 
    outputs = layers.Dense(3)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_g():
    print(''' 
        ################ MODEL ############### \n
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x) 
        outputs = layers.Dense(3)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
    ''')
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x) 
    outputs = layers.Dense(3)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_model_h():
    print(''' 
        ################ MODEL ############### \n
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(3)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
    ''')
    inputs = keras.Input(shape=(input_size, input_size, 1))
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_i():
    print(''' 
        ################ MODEL ############### \n
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(3)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
    ''')
    inputs = keras.Input(shape=(input_size, input_size, 1))
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_j():
    print(''' 
        ################ MODEL ############### \n
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
        outputs = layers.Dense(3)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
    ''')
    inputs = keras.Input(shape=(input_size, input_size, 1))
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    outputs = layers.Dense(3)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_k():
    print(''' 
        ################ MODEL ############### \n
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l1(0.01))(x)
        outputs = layers.Dense(3)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
    ''')
    inputs = keras.Input(shape=(input_size, input_size, 1))
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=11, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=5, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l1(0.01))(x)
    outputs = layers.Dense(3)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
    
model_func = eval(model_name)
model = model_func()
model.summary()

def load_model_l():
    print(''' 
        ################ MODEL ############### \n 
    filt = 16
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(inputs)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3)(x)

    ''')
    filt = 16
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(inputs)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_m():
    print(''' 
        ################ MODEL ############### \n 
    filt = 32
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=filt, kernel_size=11, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3)(x)


    ''')
    filt = 32
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=filt, kernel_size=11, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_n():
    print(''' 
        ################ MODEL ############### \n 
    filt = 64
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3)(x)


    ''')
    filt = 64
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model_o():
    print(''' 
        ################ MODEL ############### \n 
       filt = 32
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(filters=filt, kernel_size=11, activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=4)(x)
        x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
        x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(3)(x)

    ''')
    filt = 32
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(filters=filt, kernel_size=11, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=4)(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=filt, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

######################################################################    
########################## Loss functions ############################    
###################################################################### 

def abs_loss_function(y_true, y_pred):   
    abs_diff = K.abs(y_true - y_pred)
    ones = tf.ones_like(y_true)
    abs_diff_reversed = K.abs(ones - abs_diff )   
    minimum_from_two = tf.math.minimum(abs_diff, abs_diff_reversed) 
    return tf.math.reduce_mean(minimum_from_two, axis=-1)

def sqrt_abs_min_loss(y_true, y_pred):   
    abs_diff = K.abs(y_true - y_pred)
    ones = tf.ones_like(y_true)
    abs_diff_reversed = K.abs(tf.ones_like(y_true) - abs_diff )   
    minimum_from_two = tf.math.minimum(abs_diff, abs_diff_reversed) 
    min_sq = tf.math.sqrt(minimum_from_two)
    return tf.math.reduce_mean(min_sq, axis=-1) 

def smart_sqrt_abs_min_loss(y_true, y_pred):  
    punished_y_pred = tf.where((y_pred<0)|(y_pred>1), 2.0 + K.abs(y_pred), y_pred)
    abs_diff = K.abs(y_true - punished_y_pred)
    ones = tf.ones_like(y_true)
    abs_diff_reversed = K.abs(ones - abs_diff)   
    minimum_from_two = tf.math.minimum(abs_diff, abs_diff_reversed)     
    return tf.math.reduce_mean(minimum_from_two, axis=-1) 

loss_func = eval(loss_string)

######################################################################    
############## Callback function to print evaluation #################    
###################################################################### 
test_g = datagenerator(32, (input_size,input_size), test_df, 1, 3)
evaluation_list = []
accuracy_list = []
class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            m_e = self.model.evaluate(test_g, batch_size=32)
            for i in range(10):
                evaluation_list.append(m_e[0])
                accuracy_list.append(m_e[2])
            print("Loss on test data: ", m_e[0])
            for nr in range(3):
                t = test_df.values[nr][1]
                data = load_img(path = t, grayscale = True)
                data = tf.keras.utils.img_to_array(data, data_format="channels_last", dtype="float32")
                data /= 255
                data.shape = (1,) + data.shape
                X = np.asarray(data)
                print("----------{}----------".format(nr))
                euler = t.split("_")
                print("phi1", float(euler[3]))
                print("PHI",   float(euler[4]))
                print("phi2",  float(euler[5][:-4]))
                yhat = model.predict(data)
                print("predicted values", yhat*90)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


######################################################################    
##################### Compile and run the model ######################    
###################################################################### 

model.compile(optimizer = optimizer,
              loss = loss_func, 
              metrics = [loss_func, "accuracy"])  # Add run_eagerly=True to enable the numpy debugging

tg = datagenerator(32, (input_size,input_size), train_df, 1, 3)
vg = datagenerator(32, (input_size,input_size), valid_df, 1, 3)

history = model.fit(x=tg,
                    batch_size=32,
                    epochs=epochs,
                    validation_data=vg,
                    verbose=verbose,
                    callbacks=[LossHistory()])

######################################################################    
################# Predict few values for visibility ##################    
###################################################################### 

def prdict_and_print(nr):
    t = test_df.values[nr][1]
    data = load_img(path = t, grayscale = True)
    data = tf.keras.utils.img_to_array(data, data_format="channels_last", dtype="float32")
    data /= 255
    data.shape = (1,) + data.shape
    X = np.asarray(data)
    print("----------{}----------".format(nr))
    euler = t.split("_")
    print("phi1", float(euler[3]))
    print("PHI",   float(euler[4]))
    print("phi2",  float(euler[5][:-4]))
    yhat = model.predict(data)
    print("predicted values", yhat*90)

    
print("############### PREDICTIONS ###############")
for i in range(10):
    prdict_and_print(i)
print("############### PREDICTIONS ###############")

######################################################################    
######################### Evaluating model ###########################    
###################################################################### 

test_gen = datagenerator(32, (input_size, input_size), test_df, 1, 3)
results = model.evaluate(test_gen, batch_size=32)
print("test loss, test acc:", results)

######################################################################    
################## Saving accuracu and loss graph ####################    
###################################################################### 

fig = plt.figure(figsize=(12,5))

# Plot accuracy
plt.subplot(221)
plt.plot(history.history['accuracy'],'bo--', label = "acc")
plt.plot(history.history['val_accuracy'],'ro--', label = "val_accuracy")
plt.title("train_acc vs val_acc")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend()

# Plot loss function
plt.subplot(222)
plt.plot(history.history['loss'],'bo--', label = "loss")
plt.plot(history.history['val_loss'],'ro--', label = "val_loss")
plt.title("train_loss vs val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")

plt.legend()
plt.savefig("scripts/{}/Graphs/{}_Image_{}.png".format(out_path, random_id_str, unique_name))

######################################################################    
########################### Save a model #############################    
###################################################################### 

model.save("scripts/{}/Models/{}_model_{}.h5".format(out_path,random_id_str, unique_name), save_format = 'h5')


##################### extra
fig = plt.figure(figsize=(12,5))

# Plot accuracy
plt.subplot(221)
plt.plot(history.history['accuracy'],'bo--', label = "acc")
plt.plot(history.history['val_accuracy'],'ro--', label = "val_accuracy")
plt.plot(accuracy_list,'go--', label = "eval_acc")
plt.title("train_acc vs val_acc")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend()

# Plot loss function
plt.subplot(222)
plt.plot(history.history['loss'],'bo--', label = "loss")
plt.plot(history.history['val_loss'],'ro--', label = "val_loss")
plt.plot(evaluation_list,'go--', label = "eval_loss")
plt.title("train_loss vs val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")

plt.legend()
plt.savefig("scripts/{}/Graphs/eval_{}_Image_{}.png".format(out_path, random_id_str, unique_name))