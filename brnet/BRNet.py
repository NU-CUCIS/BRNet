import numpy as np
np.random.seed(1234567)
import tensorflow as tf
tf.random.set_seed(1234567)
import random
random.seed(1234567)

import re, os, csv, math, operator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import add, Add
from collections import Counter
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation
from data_utils import pre_process_data


train = pd.read_csv(r'path/to/train-set') 
val = pd.read_csv(r'path/to/validation-set') 
test = pd.read_csv(r'path/to/test-set') 

comp_col = 'pretty_comp'
prop_col = 'e_form'
input_shape = 86

new_x_train, new_y_train = pre_process_data(train, comp_col, prop_col)
new_x_val, new_y_val = pre_process_data(val, comp_col, prop_col)
new_x_test, new_y_test = pre_process_data(test, comp_col, prop_col)

in_layer = Input(shape=(input_shape,))

layer_1 = Dense(1024)(in_layer)
layer_1 = LeakyReLU()(layer_1)

fcc_1 = Dense(1024)(in_layer)
gsk_1 = add([fcc_1, layer_1])

layer_2 = Dense(1024)(gsk_1)
layer_2 = LeakyReLU()(layer_2)

gsk_2 = add([gsk_1, layer_2])


rayer_1 = Dense(1024)(in_layer)
rayer_1 = LeakyReLU()(rayer_1)

rcc_1 = Dense(1024)(in_layer)
rsk_1 = add([rcc_1, rayer_1])

rayer_2 = Dense(1024)(rsk_1)
rayer_2 = LeakyReLU()(rayer_2)

rsk_2 = add([rsk_1, rayer_2])


mayer_1 = add([gsk_2, rsk_2])


layer_5 = Dense(512)(mayer_1)
layer_5 = LeakyReLU()(layer_5)

mcc_5 = Dense(512)(mayer_1)
msk_5 = add([mcc_5, layer_5])

layer_6 = Dense(512)(msk_5)
layer_6 = LeakyReLU()(layer_6)

msk_6 = add([msk_5, layer_6])

layer_7 = Dense(512)(msk_6)
layer_7 = LeakyReLU()(layer_7)

msk_7 = add([msk_6, layer_7])

layer_8 = Dense(256)(msk_7)
layer_8 = LeakyReLU()(layer_8)

mcc_8 = Dense(256)(msk_7)
msk_8 = add([mcc_8, layer_8])

layer_9 = Dense(256)(msk_8)
layer_9 = LeakyReLU()(layer_9)

msk_9 = add([msk_8, layer_9])

layer_10 = Dense(256)(msk_9)
layer_10 = LeakyReLU()(layer_10)

msk_10 = add([msk_9, layer_10])

layer_11 = Dense(128)(msk_10)
layer_11 = LeakyReLU()(layer_11)

mcc_11 = Dense(128)(msk_10)
msk_11 = add([mcc_11, layer_11])

layer_12 = Dense(128)(msk_11)
layer_12 = LeakyReLU()(layer_12)

msk_12 = add([msk_11, layer_12])

layer_13 = Dense(128)(msk_12)
layer_13 = LeakyReLU()(layer_13)

msk_13 = add([msk_12, layer_13])

layer_14 = Dense(64)(msk_13)
layer_14 = LeakyReLU()(layer_14)

mcc_14 = Dense(64)(msk_13)
msk_14 = add([mcc_14, layer_14])

layer_15 = Dense(64)(msk_14)
layer_15 = LeakyReLU()(layer_15)

msk_15 = add([msk_14, layer_15])

layer_16 = Dense(32)(msk_15)
layer_16 = LeakyReLU()(layer_16)

mcc_16 = Dense(32)(msk_15)
msk_16 = add([mcc_16, layer_16])

out_layer = Dense(1)(msk_16)

model = Model(inputs=in_layer, outputs=out_layer)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)

# Fit the model
model.fit(new_x_train, new_y_train,verbose=2, validation_data=(new_x_val, new_y_val), epochs=3000, batch_size=32, callbacks=[es])

results = model.evaluate(new_x_test, new_y_test, batch_size=32)
print(results)

model_json = model.to_json()
with open("brnet_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("brnet_model.h5")
print("Saved model to disk")