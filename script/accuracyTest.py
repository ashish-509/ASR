from jiwer import wer

test_dataset = load_dataset("audiofolder", data_dir="data/test/")

def map_to_pred(batch):
    audio = batch["audio"]["array"]
    segments, _ = model.transcribe(audio)
    batch["pred"] = " ".join([s.text for s in segments])
    return batch

results = test_dataset.map(map_to_pred)
print(f"WER: {wer(results['text'], results['pred']):.2%}")