from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_dataset, Audio

# Load dataset
dataset = load_dataset("audiofolder", data_dir="data/")

# Initialize model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="ne", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Preprocess function
def prepare(batch):
    audio = batch["audio"]
    inputs = processor(
        audio["array"], 
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

dataset = dataset.map(prepare, batched=False, num_proc=4)

# Training arguments
args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=20,
    fp16=False,  # Enable if using CUDA
    evaluation_strategy="epoch",
    logging_dir="./logs"
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"]
)
trainer.train()
model.save_pretrained("whisper-tiny-nepali")
processor.save_pretrained("whisper-tiny-nepali")