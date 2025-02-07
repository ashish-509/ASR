import queue
import numpy as np
import sounddevice as sd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

MODEL_PATH = r"C:\Users\ashish\Desktop\ASR\script\whisper_tiny_nepali_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000  # Whisper uses 16kHz
CHUNK_DURATION = 1  # Process audio every 1 second
SILENCE_THRESHOLD = 0.005  # Adjust for noise sensitivity

print("Loading Whisper model...")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
print(f"Model loaded on {DEVICE}")

# Queue for audio chunks
audio_queue = queue.Queue()
last_transcription = ""  # To track last transcribed text

# Check if speech is present
def is_speech(audio):
    energy = np.sqrt(np.mean(audio**2))  # Compute root mean square energy
    return energy > SILENCE_THRESHOLD  # Only process if speech is detected

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Stream error: {status}")
        return

    audio = np.mean(indata, axis=1)  # Convert to mono
    if is_speech(audio):  
        audio_queue.put(audio.copy())

def transcribe():
    global last_transcription
    print("Listening... (Press Ctrl+C to exit)")
    
    while True:
        audio_chunk = audio_queue.get()
        
        # Convert to tensor for Whisper model
        input_features = processor(
            audio_chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(DEVICE)

        # Perform inference
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=50,
                num_beams=1,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="ne", task="transcribe")
            )
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        if transcription:
            romanized_text = transliterate(transcription, sanscript.DEVANAGARI, sanscript.ITRANS)
            if romanized_text != last_transcription: 
                print(f"Transcribed: {romanized_text}")
                last_transcription = romanized_text
        else:
            last_transcription = "" 

def main():
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
        ):
            transcribe()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
