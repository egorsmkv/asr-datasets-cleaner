import json
import argparse

import torch

from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from datasets import load_dataset, Audio


parser = argparse.ArgumentParser(description="Audio LID on utterances")

parser.add_argument("-f", "--file", required=True)
parser.add_argument("-t", "--to", required=True)
parser.add_argument("-m", "--model_id", required=True)
parser.add_argument("-bs", "--batch_size", type=int, required=True)
parser.add_argument("-d", "--device", required=True)
parser.add_argument("-cd", "--cache_dir", required=True)
parser.add_argument("-s", "--subset", required=True)

args = parser.parse_args()

processor = AutoFeatureExtractor.from_pretrained(args.model_id)
audio_lid = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_id).to(
    args.device
)

jsonlines = []
with open(args.file, "r") as f:
    for line in f:
        jsonlines.append(json.loads(line))

# flush the file
with open(args.to, "w") as f:
    f.write("")


ds = load_dataset("espnet/yodas2", args.subset, cache_dir=args.cache_dir)

train_set = ds["train"]
train_set = train_set.cast_column("audio", Audio(sampling_rate=16_000))

ds_iter = train_set.iter(batch_size=args.batch_size)

for i, batches in enumerate(ds_iter):
    audio_files = {}
    for audio in batches["audio"]:
        key = audio["path"].split("/")[-1].replace(".wav", "")
        audio_files[key] = torch.from_numpy(audio["array"])

    for batch in batches["utterances"]:
        if len(batch["text"]) == 0:
            continue

        utterances = []
        for j, text in enumerate(batch["text"]):
            utt_id = batch["utt_id"][j]
            start = batch["start"][j]
            end = batch["end"][j]

            audio_key = "-".join(utt_id.split("-")[:-3])
            audio_data = audio_files[audio_key]

            start_samples = int(start * 16_000)
            end_samples = int(end * 16_000)

            if start_samples > len(audio_data):
                start_samples = len(audio_data)

            extracted_audio = audio_data[start_samples:end_samples]

            utterances.append(
                {
                    "utt_id": utt_id,
                    "array": extracted_audio,
                }
            )

        for utterance in utterances:
            inputs = processor(
                utterance["array"], sampling_rate=16_000, return_tensors="pt"
            ).to(args.device)

            with torch.inference_mode():
                outputs = audio_lid(**inputs).logits

            lang_id = torch.argmax(outputs, dim=-1)[0].item()
            detected_lang = audio_lid.config.id2label[lang_id]

            print(utterance["utt_id"], detected_lang)

            with open(args.to, "a") as f_to:
                f_to.write(f"{utterance['utt_id']} {detected_lang}" + "\n")
