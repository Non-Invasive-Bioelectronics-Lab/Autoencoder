# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:02:28 2023

@author: m29244lx
"""

#%%  import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
import math
import scipy.io

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
import os


#%%  Load all preprocessed data
###  All data has the same sampling frequency = 200~Hz 
###  Data format saved are different 
EEG_clean_EOG = np.load('EEG_clean_EOG_bp.npy', allow_pickle=True)
EEG_noisy_EOG = np.load('EEG_noisy_EOG_bp.npy', allow_pickle=True)


### motion data has been propcessed in matlab: 
EEG_clean_motion = np.squeeze(np.transpose(np.load('EEG_clean_motion_bp.npy', allow_pickle=True)))
EEG_noisy_motion = np.squeeze(np.transpose(np.load('EEG_noisy_motion_bp.npy', allow_pickle=True)))

# lower SNR for synthetic signal ---in BigModel_v4 folder
# EEG_clean_EMG = np.load('EEG_clean_EMG.npy')
# EEG_noisy_EMG = np.load('EEG_noisy_EMG.npy')
# Higher SNR 
EEG_clean_EMG = np.load('EEG_clean_EMG_bp.npy')
EEG_noisy_EMG = np.load('EEG_noisy_EMG_bp.npy')


#%% calcualte SNR of contaminated dataset
import math
#Function that Calculate Root Mean Square
def rmsValue(arr):
    square = 0
    mean = 0.0
    root = 0.0
    n = len(arr)
    #Calculate square
    for i in range(0,n):
        square += (arr[i]**2)
    #Calculate Mean
    mean = (square / (float)(n))
    #Calculate Root
    root = math.sqrt(mean)
    return root


def snrValue(cleanSig, noisySig, scalingfactor):
    # Calculate SNR
    snr = 20*math.log10( rmsValue(cleanSig)/ rmsValue(noisySig*scalingfactor) )
    return snr

SNRs = []
factor=1
for i in range(0, len(EEG_clean_EMG)):
    SNRs.append(snrValue(EEG_clean_EMG[i,:], EEG_noisy_EMG[i,:], factor))
    

print("SNRs: ", SNRs)
print("SNR min: ", min(SNRs), "SNR max: ", max(SNRs))


#%%  bandpass filtering clean version data


## Before this step, it's good to plot the data from the last step 
## to double-check the signal is not filtered before, to avoid filtering twice


# fs=200
# ### bandpass filtering 
# def bandpass_filtering(data):
#     # Get nyquist frequency of signal
#     nyq = 0.5 * fs
#     # Find the normalised cut-off frequency
#     lowbound = 1/nyq  
#     highbound = 50/nyq
#     # Generate array of filter co-efficients
#     b, a = signal.butter(2, [lowbound, highbound], btype='bandpass', analog=False)
#     filtered_data = signal.filtfilt(b, a, data)
#     return filtered_data


# for i in range(len(EEG_clean_EOG)):
#     if len(EEG_clean_EOG[i])>1:
#         data = bandpass_filtering(EEG_clean_EOG[i])
#         EEG_clean_EOG[i] = data[1*fs:len(data)-1*fs]
        
#         data_n = EEG_noisy_EOG[i]
#         EEG_noisy_EOG[i] = data_n[1*fs:len(data_n)-1*fs]
        

# for i in range(len(EEG_clean_EMG)):
#     data = bandpass_filtering(EEG_clean_EMG[i])
#     EEG_clean_EMG[i] = data
   


#%% high-pass filter the Noisy EMG dataset: Reviewer's comments
fs=200
### bandpass filtering 
def highpass_filtering(data):
    # Get nyquist frequency of signal
    nyq = 0.5 * fs
    # Find the normalised cut-off frequency
    cutoff = 1/nyq  
    # Generate array of filter co-efficients
    b, a = signal.butter(2, cutoff, btype='highpass', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data



for i in range(len(EEG_noisy_EMG)):
    data = highpass_filtering(EEG_noisy_EMG[i])
    EEG_noisy_EMG[i] = data
    print(i)



#%%
fs=200
time = np.linspace(0, len(EEG_clean_EMG[0])/fs, num=len(EEG_clean_EMG[0]))

i=5
plt.subplot(211)
plt.plot(time, EEG_noisy_EMG[i], label="Corrupted EEG/EMG data",linewidth=2)
plt.ylabel('Corrupted data') 
plt.subplot(212)
plt.plot(time, EEG_clean_EMG[i], label="Ground-truth EEG/EMG data", linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Ground-truth data') 
plt.show()




#%% Segmentation    
### Data Segmentation define ###
import math
fs=200
def data_segment(data):
    segment_len = int(4*fs)  ## 4-second segment
    overlap = 0.5                        ## 50 percent overlap
    step = segment_len*(1-overlap)       ## Window step
    Num_of_segments = math.floor(len(data)/(step))-1 ## how many segments can be generated from one long recording
    segments =[];
    for i in range(Num_of_segments):
        seg = data[int(step*i):int(step*i+segment_len)]
        segments.append(seg)
        
    return segments



# Store all segments
EEG_clean_EOG_segments = []
EEG_noisy_EOG_segments = []
EEG_clean_motion_segments = []
EEG_noisy_motion_segments = []
EEG_clean_EMG_segments = []
EEG_noisy_EMG_segments = []

for i in range(0, len(EEG_clean_EOG)):
    S=[]
    S = data_segment(EEG_clean_EOG[i])
    S2 = []
    S2 = data_segment(EEG_noisy_EOG[i])
    print('Number of segments: '+str(len(S)))
    for j in range(len(S)):
        EEG_clean_EOG_segments.append(S[j])
        EEG_noisy_EOG_segments.append(S2[j])
        
        
        
for i in range(0, len(EEG_clean_motion)):
    S=[]
    S = data_segment(EEG_clean_motion[i])
    S2 = []
    S2 = data_segment(EEG_noisy_motion[i])
    print('Number of segments: '+str(len(S)))
    for j in range(len(S)):
        EEG_clean_motion_segments.append(S[j])
        EEG_noisy_motion_segments.append(S2[j])        
        

for i in range(0, len(EEG_clean_EMG)):
    S=[]
    S = data_segment(EEG_clean_EMG[i])
    S2 = []
    S2 = data_segment(EEG_noisy_EMG[i])
    print('Number of segments: '+str(len(S)))
    for j in range(len(S)):
        EEG_clean_EMG_segments.append(S[j])
        EEG_noisy_EMG_segments.append(S2[j])


#%% Normalization (segment by segment)

### EEG-EOG dataset
EEG_clean_EOG_norm = []
EEG_noisy_EOG_norm = []

maxValue_clean_EOG = []
minValue_clean_EOG = []
maxValue_noisy_EOG = []
minValue_noisy_EOG = []

### EEG-motion dataset
EEG_clean_motion_norm = []
EEG_noisy_motion_norm = []

maxValue_clean_motion = []
minValue_clean_motion = []
maxValue_noisy_motion = []
minValue_noisy_motion = []


### EEG-EMG dataset
EEG_clean_EMG_norm = []
EEG_noisy_EMG_norm = []

maxValue_clean_EMG = []
minValue_clean_EMG = []
maxValue_noisy_EMG = []
minValue_noisy_EMG = []


######################## EEG_EOG Data ################################
for i in range(0, len(EEG_clean_EOG_segments)):
    
    # clean data normalization: EOG
    data = EEG_clean_EOG_segments[i]
    data_norm = np.zeros(np.shape(data))
    
    # noisy data normalization: EOG
    data2 = EEG_noisy_EOG_segments[i]
    data2_norm = np.zeros(np.shape(data2))
    
    for j in range(0, len(data)):
        data_norm[j] = (data[j]-data.min())/(data.max()-data.min())
        data2_norm[j] = (data2[j]-data2.min())/(data2.max()-data2.min())
        
    ## record max and min values
    maxValue_clean_EOG.append(data.max())
    minValue_clean_EOG.append(data.min())
    maxValue_noisy_EOG.append(data2.max())
    minValue_noisy_EOG.append(data2.min())
    
    ## get normalizated data
    EEG_clean_EOG_norm.append(data_norm)
    EEG_noisy_EOG_norm.append(data2_norm)


################# EEG_MOTION Data ##########################
for i in range(0, len(EEG_clean_motion_segments)):
    
    # clean data normalization: EOG
    data = EEG_clean_motion_segments[i]
    data_norm = np.zeros(np.shape(data))
    
    # noisy data normalization: EOG
    data2 = EEG_noisy_motion_segments[i]
    data2_norm = np.zeros(np.shape(data2))
    
    for j in range(0, len(data)):
        data_norm[j] = (data[j]-data.min())/(data.max()-data.min())
        data2_norm[j] = (data2[j]-data2.min())/(data2.max()-data2.min())


    ## record max min values
    maxValue_clean_motion.append(data.max())
    minValue_clean_motion.append(data.min())
    maxValue_noisy_motion.append(data2.max())
    minValue_noisy_motion.append(data2.min())
    
    ## get normalizated data
    EEG_clean_motion_norm.append(data_norm.flatten())
    EEG_noisy_motion_norm.append(data2_norm.flatten())
    
    
    
################# EEG_EMG Data ##############################
for i in range(0, len(EEG_clean_EMG_segments)):
    
    # clean data normalization: EOG
    data = EEG_clean_EMG_segments[i]
    data_norm = np.zeros(np.shape(data))
    
    # noisy data normalization: EOG
    data2 = EEG_noisy_EMG_segments[i]
    data2_norm = np.zeros(np.shape(data2))
    
    for j in range(0, len(data)):
        data_norm[j] = (data[j]-data.min())/(data.max()-data.min())
        data2_norm[j] = (data2[j]-data2.min())/(data2.max()-data2.min())
        
           
    ## record max min values
    maxValue_clean_EMG.append(data.max())
    minValue_clean_EMG.append(data.min())
    maxValue_noisy_EMG.append(data2.max())
    minValue_noisy_EMG.append(data2.min())
        
    ## get normalizated data
    EEG_clean_EMG_norm.append(data_norm)
    EEG_noisy_EMG_norm.append(data2_norm)
    
    


    

#%% Training/Validation/Test Dataset Split
### 1. EEG-EOG
# clean
train_clean_EOG, testEOG = train_test_split(EEG_clean_EOG_norm,
                                          test_size=0.2,
                                          shuffle=False
                                          )

val_clean_EOG, test_clean_EOG = train_test_split(testEOG,
                                          test_size=0.5,
                                          shuffle=False
                                          )

# noisy
train_noisy_EOG, testEOG2 = train_test_split(EEG_noisy_EOG_norm,
                                          test_size=0.2,
                                          shuffle=False
                                          )

val_noisy_EOG, test_noisy_EOG = train_test_split(testEOG2,
                                          test_size=0.5,
                                          shuffle=False
                                          )




### 2. EEG-MOTION
# clean
train_clean_motion, testMotion = train_test_split(EEG_clean_motion_norm,
                                          test_size=0.2,
                                          shuffle=False
                                          )

val_clean_motion, test_clean_motion = train_test_split(testMotion,
                                          test_size=0.5,
                                          shuffle=False
                                          )

# noisy
train_noisy_motion, testMotion2 = train_test_split(EEG_noisy_motion_norm,
                                          test_size=0.2,
                                          shuffle=False
                                          )

val_noisy_motion, test_noisy_motion = train_test_split(testMotion2,
                                          test_size=0.5,
                                          shuffle=False
                                          )



### 3. EEG-EMG
# clean
train_clean_EMG, testEMG = train_test_split(EEG_clean_EMG_norm,
                                          test_size=0.2,
                                          shuffle=False
                                          )

val_clean_EMG, test_clean_EMG = train_test_split(testEMG,
                                          test_size=0.5,
                                          shuffle=False
                                          )

# noisy
train_noisy_EMG, testEMG2 = train_test_split(EEG_noisy_EMG_norm,
                                          test_size=0.2,
                                          shuffle=False
                                          )

val_noisy_EMG, test_noisy_EMG = train_test_split(testEMG2,
                                          test_size=0.5,
                                          shuffle=False
                                          )






#%% Data integration for Training and validation
train_clean = []
train_noisy = []

val_clean = []
val_noisy = []

test_clean = []
test_noisy = []


## Train set Clean
for i in range(len(train_clean_EOG)):
    train_clean.append(train_clean_EOG[i])

for i in range(len(train_clean_motion)):
    train_clean.append(train_clean_motion[i])
    
for i in range(len(train_clean_EMG)):
    train_clean.append(train_clean_EMG[i])

## Train set Noisy
for i in range(len(train_noisy_EOG)):
    train_noisy.append(train_noisy_EOG[i])

for i in range(len(train_noisy_motion)):
    train_noisy.append(train_noisy_motion[i])
    
for i in range(len(train_noisy_EMG)):
    train_noisy.append(train_noisy_EMG[i])



## Validation set Clean
for i in range(len(val_clean_EOG)):
    val_clean.append(val_clean_EOG[i])

for i in range(len(val_clean_motion)):
    val_clean.append(val_clean_motion[i])
    
for i in range(len(val_clean_EMG)):
    val_clean.append(val_clean_EMG[i])

## Validation set Noisy
for i in range(len(val_noisy_EOG)):
    val_noisy.append(val_noisy_EOG[i])

for i in range(len(val_noisy_motion)):
    val_noisy.append(val_noisy_motion[i])
    
for i in range(len(val_noisy_EMG)):
    val_noisy.append(val_noisy_EMG[i])
    
    
    
    
## Test set Clean
for i in range(len(test_clean_EOG)):
    test_clean.append(test_clean_EOG[i])

for i in range(len(test_clean_motion)):
    test_clean.append(test_clean_motion[i])
    
for i in range(len(test_clean_EMG)):
    test_clean.append(test_clean_EMG[i])

## Test set Noisy
for i in range(len(test_noisy_EOG)):
    test_noisy.append(test_noisy_EOG[i])

for i in range(len(test_noisy_motion)):
    test_noisy.append(test_noisy_motion[i])
    
for i in range(len(test_noisy_EMG)):
    test_noisy.append(test_noisy_EMG[i])    
    



#%%   
### Permute 2 lists (all_data_noisy & all_data_clean) in the same order 
import random

cc = list(zip(train_clean, train_noisy))
random.shuffle(cc)
train_clean[:],train_noisy[:]=zip(*cc) 


cc = list(zip(val_clean, val_noisy))
random.shuffle(cc)
val_clean[:],val_noisy[:]=zip(*cc) 



#%% Convert List to numpy array

x_train_clean = np.array(train_clean)
x_train_noisy = np.array(train_noisy)

x_val_clean = np.array(val_clean)
x_val_noisy = np.array(val_noisy)

x_test_clean = np.array(test_clean)
x_test_noisy = np.array(test_noisy)



#%% Count how many clean segments are included in the training set: x_train_noisy
import math
import scipy.stats

cc_trainset = []
for i in range(len(x_train_noisy)):
    cc_trainset.append(np.corrcoef(x_train_noisy[i], x_train_clean[i])[0,1])


cc_trainset = np.array(cc_trainset)

plt.hist(cc_trainset)
plt.show() 

# when cc>threshold, we think it's originally artifact free
print(sum(k>0.95 for k in cc_trainset))

#%% Count how many clean segments are included in the test set: x_test_noisy
import math
import scipy.stats

cc_testset = []
for i in range(len(x_test_noisy)):
    cc_testset.append(np.corrcoef(x_test_noisy[i], x_test_clean[i])[0,1])


cc_testset = np.array(cc_testset)

plt.hist(cc_testset)
plt.show() 

# when cc>0.95, we think it's originally artifact free
print(sum(k>0.95 for k in cc_testset))


#%% Count how many clean segments are included in the validation set: x_val_noisy
import math
import scipy.stats


cc_valset = []
for i in range(len(x_val_noisy)):
    cc_valset.append(np.corrcoef(x_val_noisy[i], x_val_clean[i])[0,1])


cc_valset = np.array(cc_valset)

plt.hist(cc_valset)
plt.show() 

# when cc>0.95, we think it's originally artifact free
print(sum(k>0.95 for k in cc_valset))




#%%
import time
# Record the start time
start_time = time.time()


######  Define a convolutional Autoencoder
class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=((800, 1))),
      layers.Conv1D(64, 3, activation='relu', padding='same', strides=1),
      layers.Conv1D(32, 3, activation='relu', padding='same', strides=1),
      layers.Conv1D(16, 3, activation='relu', padding='same', strides=1),
      layers.Conv1D(4, 3, activation='relu', padding='same', strides=1)
      ])

    self.decoder = tf.keras.Sequential([
        layers.Conv1D(16, 3, activation='relu', padding='same', strides=1),
        layers.Conv1D(32, 3, activation='relu', padding='same', strides=1),
        layers.Conv1D(64, 3, activation='relu', padding='same', strides=1),
        layers.Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')
      ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

########## set early stopping criteria
pat = 5 # the number of epochs with no improvement after which the training will stop
early_stopping =  EarlyStopping(monitor='val_loss',patience=pat, verbose=1)

########## save the model as a physical file
# model_checkpoint = ModelCheckpoint('Autoencoder_model.h5', verbose=1, save_best_only=True)
path_checkpoint = "training_1/cp.ckpt"
directory_checkpoint = os.path.dirname(path_checkpoint)
callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                 save_weights_only=True,
                                                 verbose=1)


Epochs = 1000
history = autoencoder.fit(x_train_noisy, x_train_clean,
                epochs=Epochs,
                shuffle=True,
                validation_data=(x_val_noisy, x_val_clean))


autoencoder.encoder.summary()
autoencoder.decoder.summary()

encoded_layer = autoencoder.encoder(x_test_noisy).numpy()
decoded_layer = autoencoder.decoder(encoded_layer).numpy()

# squeeze from (1712, 800, 1) to (1712, 800)
decoded_layer = np.squeeze(decoded_layer)



# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")



#%%  Plot learning curves
# Plot the learning curves 
plt.figure(figsize=(10, 5)) 
plt.subplot(1, 2, 1) 
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.title('Learning Curve - Loss') 
plt.legend() 

plt.subplot(1, 2, 2) 
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.title('Learning Curve - Accuracy') 
plt.legend() 
plt.tight_layout() 
# plt.savefig('learning_curve.pdf')
plt.show()


#%% save callbacks.History
import pickle
with open('history.pkl', 'wb') as file:
    pickle.dump(history, file)


# with open('history.pkl', 'rb') as file:
#     loaded_history = pickle.load(file)



#%% TensorFlow Lite saving and Converter

# TensorFlow --- savedModel 
# mkdir -p saved_model
autoencoder.save('saved_model/Autoencoder_revision')

# Convert the model to TensorFlow Lite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()

# Save the model.  // filename is:'model.tflite'
with open('autoencoder_revision.tflite', 'wb') as f:
  f.write(tflite_model)
  

#%%  plot the input images and Autoencoder output images
fs=200
time = np.linspace(0, len(x_test_clean[0])/fs, num=len(x_test_clean[0]))

n = 10
plt.figure(figsize=(30, 10))
for i in range(n):

    # display original 
    ax = plt.subplot(2, n, i + 1)
    plt.title("original")
    plt.plot(time, x_test_noisy[i+100,:])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("recon")
    plt.plot(time, decoded_layer[i+100,:])
    bx.get_xaxis().set_visible(True)
    bx.get_yaxis().set_visible(True)
    
plt.suptitle('Autoencoder Input and Output examples') 
plt.show()




#%% plot encoded features
fs=200
time = np.linspace(0, len(x_test_noisy[0])/fs, num=len(x_test_noisy[0]))

idx = 745
plt.figure(figsize=(20, 20))

### 1st row
ax = plt.subplot2grid(shape=(3,4), loc=(0,0), colspan=4)
plt.title("Original EEG")
plt.plot(time, x_test_noisy[idx,:])   
plt.ylabel('Amplitude [uV]')
plt.xlabel('Time [sec]')

### 2nd row
ax = plt.subplot2grid(shape=(3,4), loc=(1,0))
plt.title("encoder feature 1")
plt.plot(time, encoded_layer[idx,:,0])
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True) 

ax = plt.subplot2grid(shape=(3,4), loc=(1,1))
plt.title("encoder feature 2")
plt.plot(time, encoded_layer[idx,:,1])
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True) 

ax = plt.subplot2grid(shape=(3,4), loc=(1,2))
plt.title("encoder feature 3")
plt.plot(time, encoded_layer[idx,:,2])
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True) 

ax = plt.subplot2grid(shape=(3,4), loc=(1,3))
plt.title("encoder feature 4")
plt.plot(time, encoded_layer[idx,:,3])
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True) 

### 3rd row
ax = plt.subplot2grid(shape=(3,4), loc=(2,0), colspan=4)
plt.title("Reconstructed EEG")
plt.plot(time, decoded_layer[idx,:])  
plt.ylabel('Amplitude [uV]')
plt.xlabel('Time [sec]')
plt.ylim(0,1)




#%% results saving

# x_test_noisy :  test set (noisy signal) for autoencoder input
# x_test_clean :  ground-truth clean signal
# decoded_layer:  denoised signal

# The above 3 arrays have the same length

# EOG artifacts    :    index=[0, 372],     372 segments in total;
# Motion artifacts :    index=[372, 1002],   631 segments in total; 
# EMG artifacts    :    index=[1002, 1748],  745 segments in total;

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
np.save("x_test_noisy1", x_test_noisy, 'datasets')
np.save("x_test_clean1", x_test_clean, 'datasets')
np.save("decoded_layer1", decoded_layer, 'datasets')



#%%  Zero centered ####
## data is in range[0,1], now need to make it zero centered for statistics

z_test_noisy = np.zeros(x_test_noisy.shape)
z_test_clean = np.zeros(x_test_clean.shape)
z_decoded_layer = np.zeros(x_test_clean.shape)

### Zero_centering
for i in range(len(x_test_clean)):
    ## noisy set
    z_test_noisy[i] = x_test_noisy[i]-np.mean(x_test_noisy[i])
    ## ground truth set
    z_test_clean[i] = x_test_clean[i]-np.mean(x_test_clean[i])
    ## denoised set
    z_decoded_layer[i] = decoded_layer[i].flatten()-np.mean(decoded_layer[i].flatten())
    


#%% Reasonable Scaling: (optional)  
for i in range(len(z_test_clean)):
    adjust_factor1 = z_test_clean[i].max()/z_decoded_layer[i].max()
    adjust_factor2 = z_test_clean[i].min()/z_decoded_layer[i].min()
    
    adjust_factor = adjust_factor1
    
    z_decoded_layer[i] = z_decoded_layer[i]* adjust_factor
    

#%% CC
############ CC (Corelation Coefficients) ############
import math
import scipy.stats

CC = np.zeros(shape=(len(z_test_clean),1))
for i in range(len(z_test_clean)):
    CC[i] = np.corrcoef(z_test_clean[i], z_decoded_layer[i])[0,1]


CC_EOG = CC[0:345]
CC_motion = CC[345: 967]
CC_EMG = CC[967:1712]

plt.hist(CC_motion)
plt.show() 


#%% formular define
############### RRMSE (Relative Root Mean Square Error) ##############
#Function that Calculate Root Mean Square
def rmsValue(arr):
    square = 0
    mean = 0.0
    root = 0.0
    n = len(arr)
    #Calculate square
    for i in range(0,n):
        square += (arr[i]**2)
    #Calculate Mean
    mean = (square / (float)(n))
    #Calculate Root
    root = math.sqrt(mean)
    return root



def RRMSE(true, pred):
    ### method 1
    # num = np.sum(np.square(true - pred))
    # den = np.sum(np.square(true))
    # squared_error = num/den
    # rrmse_loss = np.sqrt(squared_error)
    
    ### method 2
    num = rmsValue(true-pred)
    den = rmsValue(true)
    rrmse_loss = num/den
    return rrmse_loss


# calcualte RMSE (Root Mean Square Error) 
def RMSE(true, pred):
    return rmsValue(true-pred)
    


#%% Evaluation without separation clean/noisy (optional)
############### RRMSE (Relative Root Mean Square Error) ##############
############### Time Domain #####################################
import math
RRMSE_timeDomain = np.zeros(shape=(len(z_test_clean),1))
for i in range(len(RRMSE_timeDomain)):
    RRMSE_timeDomain[i] = RRMSE(z_test_clean[i], z_decoded_layer[i])
    # RRMSE_timeDomain[i] = RRMSE(z_test_clean[i].astype(np.int8), z_decoded_layer[i].astype(np.int8))
    # RRMSE_timeDomain[i] = RRMSE(z_test_clean[i].astype(np.float16), z_decoded_layer[i].astype(np.float16))




RRMSE_EOG = RRMSE_timeDomain[0:345]
RRMSE_motion = RRMSE_timeDomain[345 : 967]
RRMSE_EMG = RRMSE_timeDomain[967:1712]


############### RRMSE (Relative Root Mean Square Error) ##############
############### Frequency Domain #####################################

# Calculate 'PSD' for ground truth EEG and denoised EEG
nperseg = 200
nfft=800
PSD_len= nfft//2+1

PSD_cleanEEG = np.zeros(shape=(len(z_test_clean), PSD_len))
PSD_denoisedEEG = np.zeros(shape=(len(z_test_clean), PSD_len))

for i in range(len(z_test_clean)):
    # welch default: nfft=nperseg; noverlap=nperseg//2; window='hann')
    # 1. PSD clean EEG
    f, pxx = signal.welch(z_test_clean[i], fs=200, nperseg=nperseg, nfft=nfft) 
    PSD_cleanEEG[i] = pxx
    
    # 2. PSD denoised EEG
    f, pxx = signal.welch(z_decoded_layer[i], fs=200, nperseg=nperseg, nfft=nfft) 
    PSD_denoisedEEG[i] = pxx



RRMSE_freqDomain = np.zeros(shape=(len(z_test_clean),1))
for i in range(len(RRMSE_freqDomain)):
    RRMSE_freqDomain[i] = RRMSE(PSD_cleanEEG[i], PSD_denoisedEEG[i])


RRMSE_EOGf = RRMSE_freqDomain[0:372]
RRMSE_motionf = RRMSE_freqDomain[372: 1003]
RRMSE_EMGf = RRMSE_freqDomain[1003:1748]







#%% FINAL RESULTS --- STATISTICS, without separating clean and noisy input
### The results are for EOG/MOTION/EMG dataset, with clean and noisy inputs mixed

## EOG 
print("\n EEG EOG artifacts results:")
print("RRMSE-Time: mean= ", np.mean(RRMSE_EOG), " ,std= ", np.std(RRMSE_EOG))
print("RRMSE-Freq: mean= ", np.mean(RRMSE_EOGf), " ,std= ", np.std(RRMSE_EOGf))
print("CC: mean= ", np.mean(CC_EOG), " ,std= ", np.std(CC_EOG))

## MOTION
print("\n EEG motion artifacts results:")
print("RRMSE-Time:  mean= ", np.mean(RRMSE_motion), " ,std= ", np.std(RRMSE_motion))
print("RRMSE-Freq:  mean= ", np.mean(RRMSE_motionf), " ,std= ", np.std(RRMSE_motionf))
print("CC:  mean= ", np.mean(CC_motion), " ,std= ", np.std(CC_motion))

## EMG
print("\n EEG EMG artifacts results:")
print("RRMSE-Time:  mean= ", np.mean(RRMSE_EMG), " ,std= ", np.std(RRMSE_EMG))
print("RRMSE-Freq:  mean= ", np.mean(RRMSE_EMGf), " ,std= ", np.std(RRMSE_EMGf))
print("CC:  mean= ", np.mean(CC_EMG), " ,std= ", np.std(CC_EMG))




  
#%% Load Model and saved results (when the results and model are saved, this block works)
autoencoder = tf.keras.models.load_model('saved_model/Autoencoder_CNN_model_v4')
autoencoder.summary()

encoded_layer = autoencoder.encoder(x_test_noisy).numpy()
decoded_layer = autoencoder.decoder(encoded_layer).numpy()

### Load Saved Results
x_test_noisy = np.load("x_test_noisy1.npy")
x_test_clean = np.load("x_test_clean1.npy")
decoded_layer = np.load("decoded_layer1.npy")



#%% Detect clean inputs
clean_detect = []
noisy_detect = []

CC_detectClean = np.zeros(shape=(len(z_test_clean),1))
for i in range(len(z_test_clean)):
    # calculate cc between noisy and clean version, if they are similar, it means the noisy input is clean
    # if input test_clean and test_noisy is quite different, it means the signal is noisy
    CC_detectClean[i] = np.corrcoef(z_test_clean[i], z_test_noisy[i])[0,1]
    if CC_detectClean[i]>0.95:
        clean_detect.append(i)
    else:
        noisy_detect.append(i)
        
        
### initialize the lists to store the separated data
clean_inputs = []
clean_outputs = []

noisy_inputs_EOG = []
noisy_inputs_Motion = []
noisy_inputs_EMG = []
noisy_outputs_EOG = []
noisy_outputs_Motion = []
noisy_outputs_EMG = []
ground_truth_EOG = []
ground_truth_Motion = []
ground_truth_EMG = []        

for i in range(len(clean_detect)):
    clean_inputs.append(z_test_noisy[clean_detect[i]])
    clean_outputs.append(z_decoded_layer[clean_detect[i]])
    
    
for i in range(len(noisy_detect)):
    if noisy_detect[i]<345:
        noisy_inputs_EOG.append(z_test_noisy[noisy_detect[i]])
        noisy_outputs_EOG.append(z_decoded_layer[noisy_detect[i]])
        ground_truth_EOG.append(z_test_clean[noisy_detect[i]])
    elif noisy_detect[i]>=345 and noisy_detect[i]<967:
        noisy_inputs_Motion.append(z_test_noisy[noisy_detect[i]])
        noisy_outputs_Motion.append(z_decoded_layer[noisy_detect[i]])
        ground_truth_Motion.append(z_test_clean[noisy_detect[i]])
    elif noisy_detect[i]>=967:
        noisy_inputs_EMG.append(z_test_noisy[noisy_detect[i]])
        noisy_outputs_EMG.append(z_decoded_layer[noisy_detect[i]])
        ground_truth_EMG.append(z_test_clean[noisy_detect[i]])
        

#%% SAVE THE GROUND-TRUTH clean AND reconstructed DATA, and the corrupted data to local
np.save("clean_inputs1", clean_inputs, 'datasets')
np.save("clean_outputs1", clean_outputs, 'datasets')

np.save("noisy_inputs_EOG1", noisy_inputs_EOG, 'datasets')
np.save("noisy_outputs_EOG1", noisy_outputs_EOG, 'datasets')
np.save("ground_truth_EOG1", ground_truth_EOG, 'datasets')

np.save("noisy_inputs_Motion1", noisy_inputs_Motion, 'datasets')
np.save("noisy_outputs_Motion1", noisy_outputs_Motion, 'datasets')
np.save("ground_truth_Motion1", ground_truth_Motion, 'datasets')

np.save("noisy_inputs_EMG1", noisy_inputs_EMG, 'datasets')
np.save("noisy_outputs_EMG1", noisy_outputs_EMG, 'datasets')
np.save("ground_truth_EMG1", ground_truth_EMG, 'datasets')


#%% Evaluation --- this results are used in final papers
###### Clean signal --- Reconstruction
## 1. RRMSE: time domain
clean_inputs_RRMSE=[]
clean_inputs_RRMSEABS=[]
for i in range(len(clean_inputs)):
    clean_inputs_RRMSE.append(RRMSE(clean_inputs[i], clean_outputs[i]))
    clean_inputs_RRMSEABS.append(RMSE(clean_inputs[i], clean_outputs[i]))


## 2. RRMSE: frequency domain
clean_inputs_PSD_RRMSE=[]
clean_inputs_PSD_RRMSEABS=[]

nperseg = 200
nfft=800
PSD_len= nfft//2+1

clean_inputs_PSD = np.zeros(shape=(len(clean_inputs), PSD_len))
clean_outputs_PSD = np.zeros(shape=(len(clean_inputs), PSD_len))

for i in range(len(clean_inputs)):
    # welch default: nfft=nperseg; noverlap=nperseg//2; window='hann')
    # 1. PSD input clean EEG
    f, pxx = signal.welch(clean_inputs[i], fs=200, nperseg=nperseg, nfft=nfft) 
    clean_inputs_PSD[i] = pxx
    
    # 2. PSD denoised/reconstructed EEG
    f, pxx = signal.welch(clean_outputs[i], fs=200, nperseg=nperseg, nfft=nfft) 
    clean_outputs_PSD[i] = pxx


for i in range(len(clean_inputs)):
    clean_inputs_PSD_RRMSE.append(RRMSE(clean_inputs_PSD[i], clean_outputs_PSD[i]))
    clean_inputs_PSD_RRMSEABS.append(RMSE(clean_inputs_PSD[i], clean_outputs_PSD[i]))

## 3. CC
import scipy.stats
clean_inputs_CC = []

for i in range(len(clean_inputs)):
    result = scipy.stats.pearsonr(clean_inputs[i], clean_outputs[i])    # Pearson's r
    clean_inputs_CC.append(result.statistic)




###### EEG/EOG artifacts --- Denoising
## 1. RRMSE: time domain
EOG_RRMSE=[]
EOG_RRMSEABS=[]
for i in range(len(noisy_inputs_EOG)):
    EOG_RRMSE.append(RRMSE(ground_truth_EOG[i], noisy_outputs_EOG[i]))
    EOG_RRMSEABS.append(RMSE(ground_truth_EOG[i], noisy_outputs_EOG[i]))
    
## 2. RRMSE: Frequency domain
ground_truth_EOG_PSD = np.zeros(shape=(len(noisy_inputs_EOG), PSD_len))
noisy_outputs_EOG_PSD = np.zeros(shape=(len(noisy_inputs_EOG), PSD_len))

for i in range(len(noisy_inputs_EOG)):
    # welch default: nfft=nperseg; noverlap=nperseg//2; window='hann')
    # 1. PSD input clean EEG
    f, pxx = signal.welch(ground_truth_EOG[i], fs=200, nperseg=nperseg, nfft=nfft) 
    ground_truth_EOG_PSD[i] = pxx
    
    # 2. PSD denoised/reconstructed EEG
    f, pxx = signal.welch(noisy_outputs_EOG[i], fs=200, nperseg=nperseg, nfft=nfft) 
    noisy_outputs_EOG_PSD[i] = pxx


EOG_PSD_RRMSE=[]
EOG_PSD_RRMSEABS=[]
for i in range(len(noisy_inputs_EOG)):
    EOG_PSD_RRMSE.append(RRMSE(ground_truth_EOG_PSD[i], noisy_outputs_EOG_PSD[i]))
    EOG_PSD_RRMSEABS.append(RMSE(ground_truth_EOG_PSD[i], noisy_outputs_EOG_PSD[i]))
## 3. CC
EOG_CC = []

for i in range(len(noisy_inputs_EOG)):
    result = scipy.stats.pearsonr(ground_truth_EOG[i], noisy_outputs_EOG[i])    # Pearson's r
    EOG_CC.append(result.statistic)





###### EEG/Motion artifacts --- Denoising
## 1. RRMSE: time domain
Motion_RRMSE=[]
Motion_RRMSEABS=[]
for i in range(len(noisy_inputs_Motion)):
    Motion_RRMSE.append(RRMSE(ground_truth_Motion[i], noisy_outputs_Motion[i]))
    Motion_RRMSEABS.append(RMSE(ground_truth_Motion[i], noisy_outputs_Motion[i]))
    
## 2. RRMSE: Frequency domain
ground_truth_Motion_PSD = np.zeros(shape=(len(noisy_inputs_Motion), PSD_len))
noisy_outputs_Motion_PSD = np.zeros(shape=(len(noisy_inputs_Motion), PSD_len))

for i in range(len(noisy_inputs_Motion)):
    # welch default: nfft=nperseg; noverlap=nperseg//2; window='hann')
    # 1. PSD input clean EEG
    f, pxx = signal.welch(ground_truth_Motion[i], fs=200, nperseg=nperseg, nfft=nfft) 
    ground_truth_Motion_PSD[i] = pxx
    
    # 2. PSD denoised/reconstructed EEG
    f, pxx = signal.welch(noisy_outputs_Motion[i], fs=200, nperseg=nperseg, nfft=nfft) 
    noisy_outputs_Motion_PSD[i] = pxx


Motion_PSD_RRMSE=[]
Motion_PSD_RRMSEABS=[]
for i in range(len(noisy_inputs_Motion)):
    Motion_PSD_RRMSE.append(RRMSE(ground_truth_Motion_PSD[i], noisy_outputs_Motion_PSD[i]))
    Motion_PSD_RRMSEABS.append(RMSE(ground_truth_Motion_PSD[i], noisy_outputs_Motion_PSD[i]))
## 3. CC
Motion_CC = []

for i in range(len(noisy_inputs_Motion)):
    result = scipy.stats.pearsonr(ground_truth_Motion[i], noisy_outputs_Motion[i])    # Pearson's r
    Motion_CC.append(result.statistic)




###### EEG/EMG artifacts --- Denoising
## 1. RRMSE: time domain
EMG_RRMSE=[]
EMG_RRMSEABS=[]
for i in range(len(noisy_inputs_EMG)):
    EMG_RRMSE.append(RRMSE(ground_truth_EMG[i], noisy_outputs_EMG[i]))
    EMG_RRMSEABS.append(RMSE(ground_truth_EMG[i], noisy_outputs_EMG[i]))
## 2. RRMSE: Frequency domain
ground_truth_EMG_PSD = np.zeros(shape=(len(noisy_inputs_EMG), PSD_len))
noisy_outputs_EMG_PSD = np.zeros(shape=(len(noisy_inputs_EMG), PSD_len))

for i in range(len(noisy_inputs_EMG)):
    # welch default: nfft=nperseg; noverlap=nperseg//2; window='hann')
    # 1. PSD input clean EEG
    f, pxx = signal.welch(ground_truth_EMG[i], fs=200, nperseg=nperseg, nfft=nfft) 
    ground_truth_EMG_PSD[i] = pxx
    
    # 2. PSD denoised/reconstructed EEG
    f, pxx = signal.welch(noisy_outputs_EMG[i], fs=200, nperseg=nperseg, nfft=nfft) 
    noisy_outputs_EMG_PSD[i] = pxx


EMG_PSD_RRMSE=[]
EMG_PSD_RRMSEABS=[]
for i in range(len(noisy_inputs_EMG)):
    EMG_PSD_RRMSE.append(RRMSE(ground_truth_EMG_PSD[i], noisy_outputs_EMG_PSD[i]))
    EMG_PSD_RRMSEABS.append(RMSE(ground_truth_EMG_PSD[i], noisy_outputs_EMG_PSD[i]))

## 3. CC
EMG_CC = []

for i in range(len(noisy_inputs_EMG)):
    result = scipy.stats.pearsonr(ground_truth_EMG[i], noisy_outputs_EMG[i])    # Pearson's r
    EMG_CC.append(result.statistic)





#  Convert list to numpy array
# Clean signal reconstruction
clean_inputs_RRMSE = np.array(clean_inputs_RRMSE)
clean_inputs_PSD_RRMSE = np.array(clean_inputs_PSD_RRMSE)
clean_inputs_CC = np.array(clean_inputs_CC)
# 
clean_inputs_RRMSEABS = np.array(clean_inputs_RRMSEABS)
clean_inputs_PSD_RRMSEABS = np.array(clean_inputs_PSD_RRMSEABS) 

# EOG artifacts removal
EOG_RRMSE = np.array(EOG_RRMSE)
EOG_PSD_RRMSE = np.array(EOG_PSD_RRMSE)
EOG_CC = np.array(EOG_CC)
#
EOG_RRMSEABS = np.array(EOG_RRMSEABS)
EOG_PSD_RRMSEABS = np.array(EOG_PSD_RRMSEABS)

# Motion artifacts removal
Motion_RRMSE = np.array(Motion_RRMSE)
Motion_PSD_RRMSE = np.array(Motion_PSD_RRMSE)
Motion_CC = np.array(Motion_CC)
#
Motion_RRMSEABS =np.array(Motion_RRMSEABS)
Motion_PSD_RRMSEABS = np.array(Motion_PSD_RRMSEABS)


# EMG artifacts removal
EMG_RRMSE = np.array(EMG_RRMSE)
EMG_PSD_RRMSE = np.array(EMG_PSD_RRMSE)
EMG_CC = np.array(EMG_CC)
#
EMG_RRMSEABS = np.array(EMG_RRMSEABS)
EMG_PSD_RRMSEABS = np.array(EMG_PSD_RRMSEABS)

### Print results
## Clean Input Signal ##
print("\n EEG clean input results: ")
print("RRMSE-Time: mean= ", "%.4f" % np.mean(clean_inputs_RRMSE), " ,std= ", "%.4f" % np.std(clean_inputs_RRMSE))
print("RRMSE-Freq: mean= ", "%.4f" % np.mean(clean_inputs_PSD_RRMSE), " ,std= ", "%.4f" % np.std(clean_inputs_PSD_RRMSE))
print("CC: mean= ", "%.4f" % np.mean(clean_inputs_CC), " ,std= ", "%.4f" % np.std(clean_inputs_CC))
# print("RMSE-Time: mean = ", "%.4f" % np.mean(clean_inputs_RRMSEABS), " , std= ","%.4f" % np.std(clean_inputs_RRMSEABS))
# print("RMSE-Freq: mean= ","%.4f" % np.mean(clean_inputs_PSD_RRMSEABS), " ,std= ","%.4f" % np.std(clean_inputs_PSD_RRMSEABS))

## EOG ##
print("\n EEG EOG artifacts results:")
print("RRMSE-Time: mean= ","%.4f" % np.mean(EOG_RRMSE), " ,std= ","%.4f" % np.std(EOG_RRMSE))
print("RRMSE-Freq: mean= ","%.4f" % np.mean(EOG_PSD_RRMSE), " ,std= ","%.4f" % np.std(EOG_PSD_RRMSE))
print("CC: mean= ","%.4f" % np.mean(EOG_CC), " ,std= ","%.4f" % np.std(EOG_CC))
# print("RMSE-Time: mean= ","%.4f" % np.mean(EOG_RRMSEABS), " ,std= ","%.4f" % np.std(EOG_RRMSEABS))
# print("RMSE-Freq: mean= ","%.4f" % np.mean(EOG_PSD_RRMSEABS), " ,std= ","%.4f" % np.std(EOG_PSD_RRMSEABS))

## MOTION
print(" \n EEG motion artifacts results:")
print("RRMSE-Time:  mean= ","%.4f" % np.mean(Motion_RRMSE), " ,std= ","%.4f" % np.std(Motion_RRMSE))
print("RRMSE-Freq:  mean= ","%.4f" % np.mean(Motion_PSD_RRMSE), " ,std= ","%.4f" % np.std(Motion_PSD_RRMSE))
print("CC:  mean= ","%.4f" % np.mean(Motion_CC), " ,std= ","%.4f" % np.std(Motion_CC))
# print("RMSE-Time:  mean= ","%.4f" % np.mean(Motion_RRMSEABS), " ,std= ","%.4f" % np.std(Motion_RRMSEABS))
# print("RMSE-Freq:  mean= ","%.4f" % np.mean(Motion_PSD_RRMSEABS), " ,std= ","%.4f" % np.std(Motion_PSD_RRMSEABS))

## EMG
print(" \n EEG EMG artifacts results:")
print("RRMSE-Time:  mean= ","%.4f" % np.mean(EMG_RRMSE), " ,std= ","%.4f" % np.std(EMG_RRMSE))
print("RRMSE-Freq:  mean= ","%.4f" % np.mean(EMG_PSD_RRMSE), " ,std= ","%.4f" % np.std(EMG_PSD_RRMSE))
print("CC:  mean= ", "%.4f" % np.mean(EMG_CC), " ,std= ","%.4f" % np.std(EMG_CC))
# print("RMSE-Time:  mean= ","%.4f" % np.mean(EMG_RRMSEABS), " ,std= ","%.4f" % np.std(EMG_RRMSEABS))
# print("RMSE-Freq:  mean= ","%.4f" % np.mean(EMG_PSD_RRMSEABS), " ,std= ","%.4f" % np.std(EMG_PSD_RRMSEABS))







#%% # boxplot: optional
plt.boxplot([clean_inputs_RRMSE, clean_inputs_PSD_RRMSE, clean_inputs_CC])
plt.title("Clean EEG input") 
plt.ylabel('Values')
plt.xticks([1, 2, 3], ['RRMSE-Temporal','RRMSE-Specral','CC'])


plt.boxplot([EOG_RRMSE, EOG_PSD_RRMSE, EOG_CC])
plt.title("EEG/EOG artiafcts") 
plt.ylabel('Values')
plt.xticks([1, 2, 3], ['RRMSE-Temporal','RRMSE-Specral','CC'])


plt.boxplot([Motion_RRMSE, Motion_PSD_RRMSE, Motion_CC])
plt.title("EEG/motion artiafcts") 
plt.ylabel('Values')
plt.xticks([1, 2, 3], ['RRMSE-Temporal','RRMSE-Specral','CC'])


plt.boxplot([EMG_RRMSE, EMG_PSD_RRMSE, EMG_CC])
plt.title("EEG/EMG artiafcts") 
plt.ylabel('Values')
plt.xticks([1, 2, 3], ['RRMSE-Temporal','RRMSE-Specral','CC'])



#%% Export Single example - for paper
# *********************************#
# *********************************#
fs=200
time = np.linspace(0, len(x_test_clean[0])/fs, num=len(x_test_clean[0]))

# clean: 2,5,8,25

i=25
plt.plot(time, clean_inputs[i], label="Clean EEG input",linewidth=2)
plt.plot(time, clean_outputs[i], label="Clean EEG reconstruction", color='orange', linestyle='dashed',linewidth=1)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel(r'Normalized amplitude ($\mu$V)')
plt.tight_layout()
plt.savefig('cleanEEG_single4.pdf')     
plt.show()


#%% Export single example artifacts
# EOG: 4, 45, 32, 27, 258
# Motion: 109, 156, 16,110-119
# EMG: 1, 6, 21, 29

i=110

plt.plot(time, ground_truth_Motion[i], label="Ground-truth clean EEG")
plt.plot(time, noisy_inputs_Motion[i], label="Contaminated EEG")
plt.plot(time, noisy_outputs_Motion[i], label="DAE denoised EEG",linestyle='dashed',linewidth=1.5)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel(r'Normalized amplitude ($\mu$V)')
plt.tight_layout()
plt.savefig('Motion_single4.pdf')  
plt.show()


#%% Clean Input
############### Time Domain data plot ############### 

fs=200
time = np.linspace(0, len(x_test_clean[0])/fs, num=len(x_test_clean[0]))

n = 8
plt.figure(figsize=(20,10))
for i in range(n):
    
    plt.subplot(2,4,i+1)
    plt.plot(time, clean_inputs[i+40], label="Clean EEG input",linewidth=2)
    plt.plot(time, clean_outputs[i+40], label="Clean EEG reconstruction", color='orange', linestyle='dashed',linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normalized amplitude')
    plt.rcParams.update({'font.size': 15})
    
    if i==0:
        plt.legend()
    
plt.tight_layout()    
plt.savefig('cleanEEG2.pdf')     
plt.show()

## plt.savefig() should be before plt.show(), otherwise, empty plot will be saved


#%% Noisy Input
fs=200
time = np.linspace(0, len(x_test_clean[0])/fs, num=len(x_test_clean[0]))


plt.figure(figsize=(20,10))
n=8
for i in range(n):
    plt.subplot(2,4,i+1)
    plt.plot(time, ground_truth_EMG[i+33], label="Ground-truth clean EEG")
    plt.plot(time, noisy_inputs_EMG[i+33], label="Contaminated EEG")
    plt.plot(time, noisy_outputs_EMG[i+33], label="DAE denoised EEG",linestyle='dashed',linewidth=1.5)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normalized amplitude ($\mu$V)')

plt.tight_layout()
# plt.savefig('EMGplots88.pdf')  
plt.show()
  


#%%
import matplotlib.gridspec as gridspec

# gridspec inside gridspec
fig = plt.figure(figsize=(20,10))
gs0 = gridspec.GridSpec(2, 4, figure=fig)


## idx: EOG: 1
## idx: Motion: 454
## idx: EMG: 48

idx = 48
for i in range(8):
    gs00 = gs0[i].subgridspec(3, 1)
    ax1 = fig.add_subplot(gs00[0:1,:])
    plt.plot(time, noisy_inputs_EMG[i+idx]*800, label="Contaminated EEG",color='tab:blue')
    ax1.get_xaxis().set_visible(False)
    # ax1.set_ylim([-0.6, 0.6])
    plt.ylabel(r'Amplitude ($\mu$V)')
    plt.rcParams.update({'font.size': 16})
    
    if i==0:
        plt.legend()
    
    
    ax2 = fig.add_subplot(gs00[1:3,:])
    plt.plot(time, ground_truth_EMG[i+idx], label="Ground-truth clean EEG",color='tab:green')
    plt.plot(time, noisy_outputs_EMG[i+idx], label="Reconstructed EEG",linestyle='dashed',linewidth=1.5,color = 'tab:orange')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normalized amplitude')
    plt.rcParams.update({'font.size': 16})
    
    if i==0:
        plt.legend()
    
plt.tight_layout()
plt.savefig('EMGplots4.pdf')
plt.show()





#%% plot  
### plot the input images and Autoencoder output images
fs=200
time = np.linspace(0, len(x_test_clean[0])/fs, num=len(x_test_clean[0]))

n = 10
plt.figure(figsize=(30, 10))
for i in range(n):

    # display original 
    ax = plt.subplot(3, n, i + 1)
    plt.title("original")
    plt.plot(time, x_test_noisy[i+400,:])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    

    # display reconstruction
    bx = plt.subplot(3, n, i + n + 1)
    plt.title("ground-truth")
    plt.plot(time, x_test_clean[i+400,:])
    bx.get_xaxis().set_visible(True)
    bx.get_yaxis().set_visible(True)
    
    
    cx = plt.subplot(3, n, i + n*2+ 1)
    plt.title("recon")
    plt.plot(time, decoded_layer[i+400,:])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    
plt.suptitle('Autoencoder Input and Output examples') 
plt.show()











