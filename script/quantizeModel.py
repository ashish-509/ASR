from ctranslate2.converters import TransformersConverter
import os

# Configuration
MODEL_PATH = "whisper-tiny-nepali"
OUTPUT_DIR = "whisper-tiny-ct2-int8"
QUANT_TYPE = "int8" 

# Validate model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model directory {MODEL_PATH} not found!")

\
converter = TransformersConverter(MODEL_PATH)
converter.convert(
    output_dir=OUTPUT_DIR,
    quantization=QUANT_TYPE,
    force=True,
   
    copy_files=["tokenizer.json", "preprocessor_config.json"],  # Required files
    low_cpu_memory_usage=True, 
)