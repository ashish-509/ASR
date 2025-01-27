from ctranslate2.converters import TransformersConverter

converter = TransformersConverter("whisper-tiny-nepali")
converter.convert(
    output_dir="whisper-tiny-ct2",
    quantization="int8",
    force=True
)