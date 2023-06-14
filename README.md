# Autoencoder
Autoencoder Deep Learning model for EEG artifact removal in Android smartphone

The EEG dataset (preprocessed) and the Autoencoder Python code is in "ae_model" branch.
The trained TensorFlow model, and the converted TensorFlow-Lite model are also included in "ae_model" branch.

The Android Studio project is in "android_app" branch.


Statements:
1. The EEG datasets used in this work are publicly available, and published by other researchers, the source of the dataset is provided below:
   (1). EEG/EOG artifact dataset: can be downloaded from: https://data.mendeley.com/datasets/wb6yvr725d/1; and the data usage instructions can be found in DOI:   
        https://www.sciencedirect.com/science/article/pii/S2352340916304000?via%3Dihub
   (2). EEG/Motion artifact dataset: is from PhysioNet: https://physionet.org/content/motion-artifact/1.0.0/
   (3). EEG/EMG artifact dataset: is obtained from the 'EEGDenoiseNet' paper (DOI: https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8), data can be downloaded from GitHub: 
        https://github.com/ncclabsustech/EEGdenoiseNet

   Based on these public EEG datasets, this work did appropriate signal pre-processing for the Autoencoder model development. The pre-processing details can be found in the "code_data" folder -> 
   "Readme" file.

2. The trained Autoencoder Model is saved into "saved_model" folder, after downloading the folder, set up the correct directory, and then in Python, run the following code to load the model:

    #%% Load Model from local directory
    autoencoder = tf.keras.models.load_model('saved_model/Autoencoder_CNN_model_v4')
    autoencoder.summary()
   
    #%% Run the model by using your preprocessed EEG data
    encoded_layer = autoencoder.encoder(x_test_noisy).numpy()        / x_test_noisy: your own preprocessed EEG data (the INPUT of the autoencoder model)
    decoded_layer = autoencoder.decoder(encoded_layer).numpy()       / decoded_layer: the reconstructed EEG data (the OUTPUT of the autoencoder model)

3. The "autoencoderCNN_final.tflite" is the converted autoencoder model for mobile/edge devices, used for TensorFlow-Lite. The example code for using this in Android can be found in  
   "android_app" branch

   
