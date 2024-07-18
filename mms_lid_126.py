import torch

from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from datasets import load_dataset

cache_dir = "./cache-yodas2-uk/"
ds = load_dataset("espnet/yodas2", "uk000", cache_dir=cache_dir)

model_id = "facebook/mms-lid-126"
device = "cuda:0"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(device)

print("Supported languages:")
print(list(model.config.id2label.values()))

n_samples = 10
train_iterator = iter(ds["train"])

for _ in range(n_samples):
    sample = next(train_iterator)

    audio_data = torch.from_numpy(sample["audio"]["array"])
    sample_rate = sample["audio"]["sampling_rate"]

    ids = sample["utterances"]["utt_id"]
    text = sample["utterances"]["text"]
    start = sample["utterances"]["start"]
    end = sample["utterances"]["end"]

    print("audio duration:", sample["duration"])
    print(
        "audio duration by intervals:",
        sum([end[i] - start_value for i, start_value in enumerate(start)]),
    )

    for idx, _ in enumerate(text):
        start_a = start[idx]
        end_a = end[idx]
        text_a = text[idx]

        start_samples = int(start_a * sample_rate)
        end_samples = int(end_a * sample_rate)

        if start_samples > len(audio_data):
            start_samples = len(audio_data)

        print(end_a - start_a, "||", text_a)

        extracted_audio = audio_data[start_samples:end_samples]

        inputs = processor(
            extracted_audio, sampling_rate=16_000, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs).logits

        lang_id = torch.argmax(outputs, dim=-1)[0].item()
        detected_lang = model.config.id2label[lang_id]

        print("Detected lang:", detected_lang)

    print("----" * 5)
