import os
import numpy as np
import sounddevice as sd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Define paths and device
MODEL_PATH = os.path.abspath("../whisper_tiny_nepali_final")  # Adjust the path to your model folder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the processor and model
try:
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    print(f"Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Audio parameters
SAMPLE_RATE = 16000  # Match the sample rate used during training
CHUNK_DURATION = 2   # Duration of each audio chunk in seconds

# Function to execute commands based on transcription
def execute_command(text):
    """
    Add your specific logic to execute commands based on the transcribed text.
    For example, send motor control signals or trigger events.
    """
    print(f"Executing command for detected text: {text}")

# Audio callback function for real-time processing
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Stream status error: {status}")
        return

    try:
        # Convert stereo to mono if needed and preprocess
        audio = np.mean(indata, axis=1)
        input_features = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(DEVICE)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=50, num_beams=1)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Output transcription and execute corresponding command
        print(f"Detected: {transcription}")
        execute_command(transcription)

    except Exception as e:
        print(f"Error during audio processing: {e}")

# Main function to start audio streaming
def main():
    print("Listening... (press Ctrl+C to stop)")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
        ):
            while True:
                sd.sleep(100)  # Keep the stream open
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
