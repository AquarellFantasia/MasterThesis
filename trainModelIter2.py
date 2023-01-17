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

import keras.backend as K

###################################################################### 
#################### Get arguments for script ########################
######################################################################
# sys.argv[1] - Epochs
# sys.argv[2] - optimizer
# sys.argv[3] - metrics
# sys.argv[4] - loss function name
# sys.argv[5] - image data path csv
# sys.argv[6] - verbose. 0 - no epoch logs; 1 - all logs; 2 - only epochs 
# sys.argv[7] - name to differentiate the files from one another

# load defaults
epochs      = int(sys.argv[1])
optimizer   = eval(sys.argv[2])
metrics     = eval(sys.argv[3])
loss_string = sys.argv[4]
csv_path    = sys.argv[5]
verbose     = int(sys.argv[6])
unique_name = sys.argv[7]

print("Epochs: ", sys.argv[1])
print("Optimizer: ", sys.argv[2])
print("Metrics: ", sys.argv[3])
print("Loss function name: ", sys.argv[4])
print("Csv file used: ", sys.argv[5])
print("Verbose: ", sys.argv[6])
print("Unique name: ", sys.argv[7])

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
            data = load_img(path = data_paths[i], grayscale = True)
            data = tf.keras.utils.img_to_array(data, data_format="channels_last", dtype="float32")
            data /= 255
            data.shape = (1,) + data.shape
            x[i] = np.asarray(data)
        return x.astype("float32"), np.array(y).astype("float32")
    
######################################################################    
######################## Loading the model ###########################    
###################################################################### 
print(''' ################ MODEL ############### \n 
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
model.summary()


######################################################################    
########################## Loss functions ############################    
###################################################################### 

def abs_loss_function(y_true, y_pred):   
    abs_diff = K.abs(y_true - y_pred)
    ones = tf.ones_like(y_true)
    abs_diff_reversed = K.abs(tf.ones_like(y_true) - abs_diff )   
    minimum_from_two = tf.math.minimum(abs_diff, abs_diff_reversed) 
    return K.mean(minimum_from_two)

def square_abs_min_loss(y_true, y_pred):   
    abs_diff = K.abs(y_true - y_pred)
    ones = tf.ones_like(y_true)
    abs_diff_reversed = K.abs(tf.ones_like(y_true) - abs_diff )   
    minimum_from_two = tf.math.minimum(abs_diff, abs_diff_reversed) 
    result = tf.math.square( (minimum_from_two[:, 0] + minimum_from_two[:, 1] + minimum_from_two[:, 2]) / 3 )
    return K.mean(result)

def square_abs_min_individual_loss(y_true, y_pred):   
    abs_diff = K.abs(y_true - y_pred)
    ones = tf.ones_like(y_true)
    abs_diff_reversed = K.abs(tf.ones_like(y_true) - abs_diff )   
    minimum_from_two = tf.math.minimum(abs_diff, abs_diff_reversed) 
    result = (tf.math.square(minimum_from_two[:, 0]) + 
              tf.math.square(minimum_from_two[:, 1]) + 
              tf.math.square(minimum_from_two[:, 2])) / 3 
    return K.mean(result)

loss_func = eval(loss_string)

######################################################################    
##################### Compile and run the model ######################    
###################################################################### 
model.compile(optimizer = optimizer,
              loss = loss_func, 
              metrics = metrics)  # Add run_eagerly=True to enable the numpy debugging

tg = datagenerator(32, (input_size,input_size), train_df, 1, 3)
vg = datagenerator(32, (input_size,input_size), valid_df, 1, 3)

history = model.fit(x=tg,
                    batch_size=32,
                    epochs=epochs,
                    validation_data=vg,
                    verbose=verbose)

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
plt.savefig("Graphs/{}_Image_{}.png".format(random_id_str, unique_name))

######################################################################    
######################### Evaluating models ##########################    
###################################################################### 

test_gen = datagenerator(32, (input_size, input_size), test_df, 1, 3)
results = model.evaluate(test_gen, batch_size=32)
print("test loss, test acc:", results)

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
    print("phi1", float(euler[3])*90)
    print("PHI",   float(euler[4])*90)
    print("phi2",  float(euler[5][:-4])*90)
    yhat = model.predict(data)
    print("predicted values", yhat*90)

    
print("############### PREDICTIONS ###############")
for i in range(10):
    prdict_and_print(i)
print("############### PREDICTIONS ###############")

######################################################################    
########################### Save a model #############################    
###################################################################### 

model.save("Models/{}_model_{}.h5".format(random_id_str, unique_name), save_format = 'h5')