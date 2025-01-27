from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

# Load model (works on Windows/Linux)
model = WhisperModel(
    "whisper-tiny-ct2",
    device="cpu",  # Use "cuda" if GPU is available
    compute_type="int8"
)

# Command mapping
COMMAND_MAP = {
    "सहारा अगाडि": "forward",
    "सहारा पछाडि": "backward",
    "सहारा दायाँ": "right",
    "सहारा बायाँ": "left",
    "सहारा रोक": "stop"
}

def execute_command(text: str):
    for cmd, action in COMMAND_MAP.items():
        if cmd in text:
            print(f"Action: {action}")
            # Add motor control logic here (e.g., serial communication)
            break

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 2)  # 2-second chunks

def audio_callback(indata, frames, time, status):
    audio = indata[:, 0].astype(np.float32)  # Convert to mono
    segments, _ = model.transcribe(
        audio,
        vad_filter=True,
        language="ne",
        beam_size=2  # Faster decoding
    )
    for segment in segments:
        execute_command(segment.text)

# Start streaming
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=CHUNK_SIZE
):
    print("Listening... (press Ctrl+C to stop)")
    while True:
        pass  # Runs indefinitely