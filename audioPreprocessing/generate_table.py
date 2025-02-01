import os
import librosa
import numpy as np  
from tabulate import tabulate

def generate_table(cleaned_dir):
    headers = ["File Name", "Sample Rate (Hz)", "Duration (s)", "Intensity (dB)", "File Path"]
    file_info = []

    for root, _, files in os.walk(cleaned_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    # Load audio
                    audio, sr = librosa.load(file_path, sr=None)
                    duration = librosa.get_duration(y=audio, sr=sr)
                    
                    # Calculate intensity
                    rms = np.sqrt(np.mean(audio**2))
                    intensity = 20 * np.log10(rms)
                    
                    # Add info to the table
                    file_info.append([file, sr, f"{duration:.2f}", f"{intensity:.2f}", file_path])
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    # Print table
    print(tabulate(file_info, headers=headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
    cleaned_dir = "../data"
    generate_table(cleaned_dir)
