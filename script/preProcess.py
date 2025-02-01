import subprocess
from pathlib import Path

def process_audio(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # output_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


    for wav_file in input_dir.glob("*.wav"):
        cmd = [
            "sox",
            str(wav_file),
            str(output_dir / wav_file.name),
            "silence", "1", "0.1", "1%", "reverse", 
            "silence", "1", "0.1", "1%", "reverse",
            "gain", "-n", "-3"
        ]
        subprocess.run(cmd, check=True)

# Process all training data
process_audio("../data/train", "../data/train")
