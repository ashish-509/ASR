import os
import torch
import matplotlib.pyplot as plt
from datasets import DatasetDict, Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_audio_text_pairs(directory):
    """Load audio files with Windows path compatibility"""
    text_mapping = {
        'A': 'सहारा अगाडि',
        'P': 'सहारा पछाडि',
        'D': 'सहारा दायाँ',
        'B': 'सहारा बायाँ',
        'R': 'सहारा रोक'
    }
    
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            prefix = filename[0].upper()
            if prefix in text_mapping:
                audio_path = os.path.abspath(os.path.join(directory, filename))
                data.append({
                    "audio": audio_path.replace("\\", "/"),  # Force Unix-style paths
                    "text": text_mapping[prefix]
                })
    return data

def plot_training_metrics(trainer):
    """Plot training and validation metrics from trainer's state"""
    history = trainer.state.log_history
    
    # Extract metrics
    train_loss = [x['loss'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    eval_steps = [x['step'] for x in history if 'eval_loss' in x]
    
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Progress')
    plt.grid(True)
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(eval_steps, eval_loss, label='Validation Loss', color='orange')
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Loss')
    plt.title('Validation Loss Progress')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def main():
    # Load data
    train_data = load_audio_text_pairs("../data/train")
    validation_data = load_audio_text_pairs("../data/validation")

    # Create dataset
    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "audio": [x["audio"] for x in train_data],
            "text": [x["text"] for x in train_data]
        }),
        "validation": Dataset.from_dict({
            "audio": [x["audio"] for x in validation_data],
            "text": [x["text"] for x in validation_data]
        })
    }).cast_column("audio", Audio(sampling_rate=16000))

    # Initialize components
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="ne")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ne", task="transcribe")

    def prepare_dataset(batch):
        """Single-process preparation for Windows compatibility"""
        audio = batch["audio"]
        inputs = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            padding="max_length",
            max_length=30 * 16000
        )
        labels = processor.tokenizer(
            batch["text"], 
            padding="max_length", 
            max_length=128
        ).input_ids
        return {
            "input_features": inputs.input_features[0].numpy(),
            "labels": labels
        }

    # Process dataset without multiprocessing
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=1  # Disable multiprocessing
    )

    # Training setup
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper_tiny_nepali",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=1000,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=25,
        predict_with_generate=True,
        report_to="none",
        save_steps=100,
        dataloader_num_workers=0,  # Disable multiprocess data loading
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False
    )

    # Data collator
    def data_collator(features):
        return {
            "input_features": torch.stack([torch.tensor(f["input_features"]) for f in features]),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(f["labels"]) for f in features],
                batch_first=True,
                padding_value=processor.tokenizer.pad_token_id
            )
        }

    # Metrics calculation
    wer_metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Generate and save plots
    plot_training_metrics(trainer)

    # Save final model
    trainer.save_model("whisper_tiny_nepali_final")
    processor.save_pretrained("whisper_tiny_nepali_final")

    print("\nTraining complete!")
    print("Training metrics saved to: training_metrics.png")
    print("Final model saved to: whisper_tiny_nepali_final")

if __name__ == "__main__":
    main()