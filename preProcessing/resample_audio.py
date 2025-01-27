import os
import librosa
import soundfile as sf

def resample_audio(cleaned_dir, target_sampling_rate=16000):
    for root, _, files in os.walk(cleaned_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    # Load audio
                    audio, original_sr = librosa.load(file_path, sr=None)
                    
                    # Resample if necessary
                    if original_sr != target_sampling_rate:
                        resampled_audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sampling_rate)
                        sf.write(file_path, resampled_audio, target_sampling_rate)
                        # print(f"Resampled {file} from {original_sr} Hz to {target_sampling_rate} Hz")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    cleaned_dir = "../dataset/cleaned"
    resample_audio(cleaned_dir)