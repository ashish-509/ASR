import os
import librosa
import soundfile as sf
import numpy as np  

def normalize_intensity(cleaned_dir, target_intensity=-20.0):
    for root, _, files in os.walk(cleaned_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    # Load audio
                    audio, sr = librosa.load(file_path, sr=None)
                    
                    # Calculate current intensity in dB
                    rms = np.sqrt(np.mean(audio**2))
                    current_intensity = 20 * np.log10(rms)
                    
                    # Calculate scaling factor
                    scaling_factor = 10 ** ((target_intensity - current_intensity) / 20)
                    
                    # Normalize intensity
                    normalized_audio = audio * scaling_factor
                    sf.write(file_path, normalized_audio, sr)
                    print(f"Normalized intensity of {file} to {target_intensity} dB")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    cleaned_dir = "../data"
    normalize_intensity(cleaned_dir)
