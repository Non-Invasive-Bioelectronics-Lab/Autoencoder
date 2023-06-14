Autoencoder Training:

'BigModel_v4.py' --- The complete Python script for the autoencoder model development, it contains: 
(1). EEG data loading;  
(2). Data segmentation;  
(3). Data normalization;  
(4). EOG/motion/EMG artefact dataset integration and sorting (in the same order for ground-truth data and corrupted data);
(5). Train/Validation/Test set split;
(6). Model training;
(7). Model Evaluation;
(8). Saving model, loading model, converting model, saving results, plotting results, etc.


The datasets here are the saved EOG, motion, EMG artifacts datasets after preprocessing. Note: The dataset provided here is before segmentation and normalization, etc.

Dataset description:

EOG artifacts:
1. EEG_clean_EOG_bp: raw dataset -> 1-50~Hz bandpass filtering -> cut out 2-seconds epochs in the begining and the end. 
2. EEG_noisy_EOG_bp: raw dataset -> no filtering -> cut out 2-seconds epochs in the begining and the end.

Motion artifacts:
1. EEG_clean_motion_matlab: raw dataset -> detrend -> downsample to 200~Hz -> 1-50~Hz bandpass filtering -> cut out 5-seconds epochs in the begining and the end.
2. EEG_noisy_motion_matlab: raw dataset -> detrend -> downsample to 200~Hz -> no filtering -> cut out 5-seconds epochs in the begining and the end.

EMG artifacts:
1. EEG_clean_EMG_bp: raw dataset -> 1-50~Hz bandpass filtering 
2. EEG_noisy_EMG_bp: raw dataset -> no filtering 
