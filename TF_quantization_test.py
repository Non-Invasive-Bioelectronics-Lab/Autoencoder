# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:47:54 2024

@author: m29244lx
"""

# Autoencoder model quantization tests

#%%  import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
import math
import scipy.io
from tensorflow.keras.models import Model



#%% load DAE model
autoencoder = tf.keras.models.load_model('C:/Users/m29244lx/OneDrive - The University of Manchester/Year3/code/FinalModel/Autoencoder_pre_bp/saved_results/saved_model/Autoencoder_revision')
autoencoder.summary()

#%% Post-training quantization: int8 conversion

## Dynamic range quantization is a recommended starting point 
## because it provides reduced memory usage and faster computation 
## without you having to provide a representative dataset for calibration. 
## This type of quantization, statically quantizes only the weights 
## from floating point to integer at conversion time, which provides 8-bits of precision:
    
    
## Dynamic range quantization   
converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/m29244lx/OneDrive - The University of Manchester/Year3/code/FinalModel/Autoencoder_pre_bp/saved_results/saved_model/Autoencoder_revision')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_int8 = converter.convert()

# Save the model. 
with open('autoencoder_int8.tflite', 'wb') as f:
  f.write(tflite_model_int8)


#%% Post-training quantization: float16 conversion
converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/m29244lx/OneDrive - The University of Manchester/Year3/code/FinalModel/Autoencoder_pre_bp/saved_results/saved_model/Autoencoder_revision')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model_float16 = converter.convert()

# Save the model.   
with open('autoencoder_float16.tflite', 'wb') as f:
  f.write(tflite_model_float16)




#%% Load saved data
x_test_noisy = np.load("C:/Users/m29244lx/OneDrive - The University of Manchester/Year3/code/FinalModel/Autoencoder_pre_bp/saved_results/x_test_noisy1.npy")
x_test_clean = np.load("C:/Users/m29244lx/OneDrive - The University of Manchester/Year3/code/FinalModel/Autoencoder_pre_bp/saved_results/x_test_clean1.npy")


#%% 1. Run TesnorFlow model
autoencoder = tf.keras.models.load_model('C:/Users/m29244lx/OneDrive - The University of Manchester/Year3/code/FinalModel/Autoencoder_pre_bp/saved_results/saved_model/Autoencoder_revision')
autoencoder.summary()

encoded_layer = autoencoder.encoder(x_test_noisy).numpy()
decoded_layer = autoencoder.decoder(encoded_layer).numpy()



#%% 2. Run tensorflow lite models
def runTFLite(input_data):

    # Load the TFLite model
    # Option 1: 'autoencoder_revision.tflite'
    # Option 2: 'autoencoder_int8.tflite'
    # Option 1: 'autoencoder_float16.tflite'
    
    interpreter = tf.lite.Interpreter(model_path='autoencoder_float16.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare the test dataset (replace with your test data)
    test_data = input_data.astype(np.float32)
    
    
    # Run inference on each test sample
    results = []
    for sample in test_data:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], sample.reshape((1, 800)))
    
        # Run inference
        interpreter.invoke()
    
        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results.append(output_data)
    
    # Convert the results to a NumPy array
    results = np.array(results)
    results = np.squeeze(results, axis=(1,3))

    return results



decoded_layer = runTFLite(x_test_noisy)




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
    





#%% Evaluation --- this metrics results are used in final papers --- Just run
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

## EOG ##
print("\n EEG EOG artifacts results:")
print("RRMSE-Time: mean= ","%.4f" % np.mean(EOG_RRMSE), " ,std= ","%.4f" % np.std(EOG_RRMSE))
print("RRMSE-Freq: mean= ","%.4f" % np.mean(EOG_PSD_RRMSE), " ,std= ","%.4f" % np.std(EOG_PSD_RRMSE))
print("CC: mean= ","%.4f" % np.mean(EOG_CC), " ,std= ","%.4f" % np.std(EOG_CC))

## MOTION
print(" \n EEG motion artifacts results:")
print("RRMSE-Time:  mean= ","%.4f" % np.mean(Motion_RRMSE), " ,std= ","%.4f" % np.std(Motion_RRMSE))
print("RRMSE-Freq:  mean= ","%.4f" % np.mean(Motion_PSD_RRMSE), " ,std= ","%.4f" % np.std(Motion_PSD_RRMSE))
print("CC:  mean= ","%.4f" % np.mean(Motion_CC), " ,std= ","%.4f" % np.std(Motion_CC))

## EMG
print(" \n EEG EMG artifacts results:")
print("RRMSE-Time:  mean= ","%.4f" % np.mean(EMG_RRMSE), " ,std= ","%.4f" % np.std(EMG_RRMSE))
print("RRMSE-Freq:  mean= ","%.4f" % np.mean(EMG_PSD_RRMSE), " ,std= ","%.4f" % np.std(EMG_PSD_RRMSE))
print("CC:  mean= ", "%.4f" % np.mean(EMG_CC), " ,std= ","%.4f" % np.std(EMG_CC))
























