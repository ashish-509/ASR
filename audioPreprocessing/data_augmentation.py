import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf

base_dir = "../data"  
output_dir = "../data"  

# Create output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

def augment_audio(file_path, output_path):
    """Apply augmentations to an audio file and save the result."""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Augmentations :
        
        # 1. Time stretching
        y_stretched = librosa.effects.time_stretch(y, rate=0.9)
        
        # 2. Pitch shifting
        y_pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        
        # 3. Add noise
        noise = np.random.normal(0, 0.005, y.shape)
        y_noisy = y + noise

        # 4. Time compressing
        y_compressed = librosa.effects.time_stretch(y, rate=1.1)
        
        # Save augmented data
        augmented_files = {
            "stretched.wav": y_stretched,
            "pitch_shifted.wav": y_pitch_shifted,
            "noisy.wav": y_noisy,
            "compressed.wav": y_compressed,
        }
        
        for suffix, augmented_audio in augmented_files.items():
            augmented_file_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(file_path))[0]}_{suffix}")
            sf.write(augmented_file_path, augmented_audio, sr)
        
        print(f"Augmented files saved for: {file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):  # Process only audio files
                input_path = os.path.join(root, file)
                
                # Determine relative output path
                relative_path = os.path.relpath(root, input_dir)
                target_dir = os.path.join(output_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)
                
                augment_audio(input_path, target_dir)

# Process all files in the cleaned directory
process_directory(base_dir, output_dir)

print(f"Augmented data saved in: {output_dir}")
