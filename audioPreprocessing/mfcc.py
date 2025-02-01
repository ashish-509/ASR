import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_mfcc(file_path, n_mfcc=40):
   
    # Load the audio file
    y, sr = librosa.load(file_path, sr=16000)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Compute the mean MFCC vector
    mean_mfcc = np.mean(mfcc.T, axis=0)
    return mean_mfcc, mfcc

def visualize_mfcc(mfcc, file_name, sr=16000):
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=512, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'MFCC Visualization for {file_name}')
    plt.tight_layout()
    plt.show()


# Main script to process all files

def process_audio_files(parent_dir):
   
    for root, _, files in os.walk(parent_dir):
        for file_name in files:
            if file_name.endswith('.wav'):  # Process only .wav files
                file_path = os.path.join(root, file_name)
                print(f"Processing: {file_path}")
                
                # Extract MFCC
                mean_mfcc, mfcc = extract_mfcc(file_path)
                print(f"Mean MFCC for {file_name}:\n{mean_mfcc}\n")
                
                # Visualize MFCC
                
                visualize_mfcc(mfcc, file_name)

# Path to the parent directory containing subfolders
parent_dir = '../data'

# Call the processing function
process_audio_files(parent_dir)
