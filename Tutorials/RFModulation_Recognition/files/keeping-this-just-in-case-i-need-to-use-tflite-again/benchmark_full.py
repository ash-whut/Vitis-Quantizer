from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras
from tensorflow import Tensor
print("Tensorflow version is ", tf.__version__)
print('Keras version      : ',keras.__version__)
import numpy as np
import os, sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import h5py as h5
from sklearn.metrics import classification_report, confusion_matrix
import random
import time
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import  Dropout, Activation, GlobalAveragePooling1D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.layers import Reshape, Dense, Flatten, Add
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, History, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from ai_edge_litert.interpreter import Interpreter

from random import shuffle
import gc
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import multiprocessing

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["OMP_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {0}) 
## Debugging - Using CPU for inference for temporary benchmarking - Comment out if needed
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

data_file = '/home/ashwin/datasets/RADIOML_2021_07_INT8/RADIOML_2021_07_INT8.hdf5'
file_handle = h5.File(data_file,'r+')

myData = file_handle['X'][:]  #1024x2 samples 
myMods = file_handle['Y'][:]  #mods 
mySNRs = file_handle['Z'][:]  #snrs  

print(np.shape(myData))
print(np.shape(myMods))
print(np.shape(mySNRs))
file_handle.close()

np.random.seed(0)

snrs = list(np.unique(mySNRs.T[0]))  
print(snrs)

mods = ["OOK","4ASK","8ASK",
        "BPSK","QPSK","8PSK","16PSK","32PSK",
        "16APSK","32APSK","64APSK","128APSK",
        "16QAM","32QAM","64QAM","128QAM","256QAM",
        "AM-SSB-WC","AM-SSB-SC","AM-DSB-WC","AM-DSB-SC","FM",
        "GMSK","OQPSK","BFSK","4FSK","8FSK"]

num_classes = np.shape(mods)[0]
print("The number of classes is ", num_classes)

myData = myData.reshape(myData.shape[0], 1024, 1, 2) 

X_train ,X_test ,Y_train ,Y_test, Z_train, Z_test = train_test_split(myData, myMods, mySNRs, test_size=0.2, random_state=0)
X_val, X_test, Y_val, Y_test, Z_val, Z_test = train_test_split(X_test, Y_test, Z_test, test_size=0.5, random_state=0)
print (np.shape(X_test))
print (np.shape(Y_test))
print (np.shape(Z_test))
print (np.shape(X_train))
print (np.shape(Y_train))
print (np.shape(Z_train))
del myData, myMods, mySNRs

def best_snr_data(X, Y, Z):
    best_snr_indices = np.where(np.any(Z == 30, axis=1))
    return X[best_snr_indices], Y[best_snr_indices]

X_best_test, Y_best_test = best_snr_data(X_test, Y_test, Z_test)
print(np.shape(X_best_test))
print(np.shape(Y_best_test))

def usable_snr_data(X, Y, Z):
    best_snr_indices = np.where(np.any(Z >= 6, axis=1))
    return X[best_snr_indices], Y[best_snr_indices]

X_usable_test, Y_usable_test = usable_snr_data(X_test, Y_test, Z_test)
print(np.shape(X_usable_test))
print(np.shape(Y_usable_test))

input_shp = list(X_train.shape[1:])
print("Dataset Shape={0} CNN Model Input layer={1}".format(X_train.shape, input_shp))
classes = mods

model= tf.keras.models.load_model('resnet_checkpoints/model_0.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()

quantized_model_save_path = 'resnet_checkpoints/model_0_fp16.tflite'

if os.path.isfile(quantized_model_save_path) == False:
    with open(quantized_model_save_path, 'wb') as f:
        f.write(tflite_quant_model)

def evaluate_batch(test_samples, test_labels, model_path):
    predictions = []
    model = tf.keras.models.load_model(model_path)
    for test_sample in test_samples:
        output = model.predict(test_sample)
        predicted_label = np.argmax(output[0])
        predictions.append(predicted_label)

    return predictions

def evaluate(model_path, test_samples, test_labels, num_workers):
    start_time = time.time()

    # Computing number of samples used by each worker
    batch_size = int(np.ceil(np.shape(test_samples)[0] / num_workers))
    batches = []
    for i in range(0, np.shape(test_samples)[0], batch_size):
        batch_samples = test_samples[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        batches.append((batch_samples, batch_labels, model_path))

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(evaluate_batch, batches)

    # Flatten all predictions
    all_predictions = np.concatenate(results)

    # Truncate ground truth labels if uneven chunking
    trimmed_labels = test_labels[:len(all_predictions)]
    accuracy = (all_predictions == np.argmax(trimmed_labels, axis=1)).mean()

    end_time = time.time()
    print(f"\nTotal evaluation time (parallel): {end_time - start_time:.2f} seconds")
    return accuracy

test_accuracy_full_model_all_snrs = evaluate('resnet_checkpoints/model_0.h5', X_test, Y_test, 1)
print(f"\nTotal evaluation time (parallel): {end_time - start_time:.2f} seconds")
print(test_accuracy_full_model_all_snrs)

