import os
import librosa
import soundfile as sf
from pathlib import Path

def resample_audio(cleaned_dir, target_sampling_rate=16000):
    cleaned_dir = Path(cleaned_dir) 
    if not cleaned_dir.is_dir():
        print(f"Provided path '{cleaned_dir}' is not a valid directory.")
        return

    for file_path in cleaned_dir.rglob("*.wav"):  # Recursively find all .wav files
        try:
            audio, original_sr = librosa.load(file_path, sr=None)
            
            # Resample if necessary
            if original_sr != target_sampling_rate:
                resampled_audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sampling_rate)
                sf.write(file_path, resampled_audio, target_sampling_rate)
                print(f"Resampled '{file_path}' from {original_sr} Hz to {target_sampling_rate} Hz")
            else:
                print(f"'{file_path}' is already at {target_sampling_rate} Hz")
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")

if __name__ == "__main__":
    cleaned_dir = r"..\data" 
    resample_audio(cleaned_dir)
