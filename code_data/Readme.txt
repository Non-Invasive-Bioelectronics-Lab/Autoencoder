Autoencoder Training:

'BigModel_v4.py' --- The Python script for training the Autoencoder for EEG artifact removal


The datasets here are the saved EOG, motion, EMG artifacts datasets after preprocessing
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