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
    
    return [
        {
            "audio": os.path.abspath(os.path.join(directory, f)).replace("\\", "/"),
            "text": text_mapping[f[0].upper()]
        }
        for f in os.listdir(directory)
        if f.endswith(".wav") and f[0].upper() in text_mapping
    ]

def plot_training_metrics(trainer):
    """Plot combined training and validation loss curves"""
    history = trainer.state.log_history
    
    # Extract metrics with actual step numbers
    train_steps = [x['step'] for x in history if 'loss' in x]
    train_loss = [x['loss'] for x in history if 'loss' in x]
    eval_steps = [x['step'] for x in history if 'eval_loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    
    plt.figure(figsize=(10, 6))
    
    # Plot both curves
    plt.plot(train_steps, train_loss, label='Training Loss', alpha=0.8)
    plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o', linestyle='--', alpha=0.8)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and prepare datasets
    def create_dataset(split):
        return Dataset.from_dict({
            "audio": [x["audio"] for x in load_audio_text_pairs(f"../data/{split}")],
            "text": [x["text"] for x in load_audio_text_pairs(f"../data/{split}")]
        })
    
    dataset = DatasetDict({
        "train": create_dataset("train"),
        "validation": create_dataset("validation")
    }).cast_column("audio", Audio(sampling_rate=16000))

    # Initialize model components
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="ne")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ne", task="transcribe")

    # Preprocessing function
    def prepare_dataset(batch):
        audio = batch["audio"]
        features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            padding="max_length",
            max_length=30 * 16000
        )
        return {
            "input_features": features.input_features[0].numpy(),
            "labels": processor.tokenizer(
                batch["text"], 
                padding="max_length", 
                max_length=128
            ).input_ids
        }

    # Process datasets
    for split in ["train", "validation"]:
        dataset[split] = dataset[split].map(
            prepare_dataset,
            remove_columns=dataset[split].column_names,
            num_proc=1
        )

    # Training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper_tiny_nepali",
        per_device_train_batch_size=16,  # Increased batch size
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,    # Better GPU utilization
        learning_rate=1e-4,
        warmup_steps=50,
        max_steps=500,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=50,                # Less frequent logging
        predict_with_generate=True,
        report_to="none",
        save_steps=200,                  # Less frequent saving
        dataloader_num_workers=0,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        gradient_checkpointing=True      # Memory optimization
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
    # def compute_metrics(pred):
    #     pred_ids = pred.predictions
    #     label_ids = torch.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        
    #     pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    #     label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
    #     return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


    def compute_metrics(pred):
        # Convert numpy arrays to PyTorch tensors first
        label_ids = torch.from_numpy(pred.label_ids)
    
        # Replace -100 with pad token ID
        label_ids = torch.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
    
        # Decode predictions and labels
        pred_str = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


    # Initialize and run training
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor
    )

    print("Starting training...")
    trainer.train()

    # Save outputs
    plot_training_metrics(trainer)
    trainer.save_model("whisper_tiny_nepali_final_2")
    processor.save_pretrained("whisper_tiny_nepali_fina_2l")

    print("\nTraining complete!")
    print("Loss curve saved: training_validation_loss.png")
    print("Model saved: whisper_tiny_nepali_final")

if __name__ == "__main__":
    main()