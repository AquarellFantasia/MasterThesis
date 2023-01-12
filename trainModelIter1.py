#################### Imports ########################
import sys
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.utils import Sequence
from keras.utils import load_img

w,h = 800,600

df = pd.read_csv('lables.csv')
train_df = df[df['ImagePath'].str.contains("train")]
test_df = df[df['ImagePath'].str.contains("test")]
valid_df = df[df['ImagePath'].str.contains("valid")]

#########################################################
################## Data generator #######################
#########################################################
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

#################### Data loader ########################
    def __dataloader(self, 
            img_size,
            data_paths,
            batch_params_paths,
            input_channels,
            output_channels):
        x = np.zeros((len(data_paths), img_size[0], img_size[1], input_channels))
        y = batch_params_paths
        
        
        for i in range(len(data_paths)):
            data = load_img(path = data_paths[i], grayscale = True)
            data = tf.keras.utils.img_to_array(data, data_format="channels_last", dtype="float32")
            data /= 255
            data.shape = (1,) + data.shape
            x[i] = np.asarray(data)
        return np.array(x).astype("float32"), np.array(y).astype("float32")
    
#################### Model ########################    
inputs = keras.Input(shape=(600, 800, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(3, activation="sigmoid")(x)

model_no_max_pool = keras.Model(inputs=inputs, outputs=outputs)
model_no_max_pool.summary()
    
#################### Loss functions ########################
def custom_loss_function(y_true, y_pred):
    # Calculate the error for each parameter using sqrt(2 - cos(y_hat - y_pred))
    param_1_error = tf.math.sqrt(2 - tf.math.cos(y_true[:, 0] - y_pred[:, 0]))
    param_2_error = tf.math.sqrt(2 - tf.math.cos(y_true[:, 1] - y_pred[:, 1]))
    param_3_error = tf.math.sqrt(2 - tf.math.cos(y_true[:, 2] - y_pred[:, 2]))

    # Calculate the total loss as the mean of the errors for each parameter
    total_loss = (param_1_error + param_2_error + param_3_error) / 3
    
    return total_loss

#################### train model ########################    
    
model_no_max_pool.compile(optimizer='adam', loss="mean_squared_error")
tg = datagenerator(32, (600,800), train_df, 1, 3)
vg = datagenerator(32, (600,800), valid_df, 1, 3)

history = model_no_max_pool.fit(x=tg,
                    batch_size=32,
                    epochs=20,
                    verbose=2,
                    validation_data=vg)


#################### Print prediction ########################

t = test_df.values[4][1]
data = load_img(path = t, grayscale = True)
data = tf.keras.utils.img_to_array(data, data_format="channels_last", dtype="float32")
data /= 255
data.shape = (1,) + data.shape
X = np.asarray(data)

yhat = model_no_max_pool.predict(data)
print(t)
print(yhat*90)