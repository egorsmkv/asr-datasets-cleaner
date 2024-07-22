import os
import json
import argparse

import librosa
import torch
import torchaudio

from datasets import load_dataset, Audio

os.environ["HF_DATASETS_OFFLINE"] = "true"

parser = argparse.ArgumentParser(description="Extract WAV utterances in 16 kHz")

parser.add_argument("-dd", "--dataset_dir", required=True)
parser.add_argument("-ss", "--subset", required=True)
parser.add_argument("-cd", "--cache_dir", required=True)
parser.add_argument("-wd", "--wav_dir", required=True)
parser.add_argument("-t", "--to", required=True)
parser.add_argument("-bs", "--batch_size", type=int, required=True)

args = parser.parse_args()

wav_dir = args.wav_dir
subset = args.subset
data_dir = f"data/{subset}"
sampling_rate = 16_000

# flush the file
with open(args.to, "w") as f:
    f.write("")

ds = load_dataset(
    args.dataset_dir,
    subset,
    data_dir=data_dir,
    trust_remote_code=True,
    cache_dir=args.cache_dir,
)

train_set = ds["train"]
train_set = train_set.cast_column(
    "audio", Audio(sampling_rate=sampling_rate, mono=True)
)

ds_iter = train_set.iter(batch_size=args.batch_size)

for batch in ds_iter:
    audio_files = {}
    for audio in batch["audio"]:
        key = audio["path"].split("/")[-1].replace(".wav", "")
        audio_files[key] = torch.from_numpy(audio["array"])

    for utterances in batch["utterances"]:
        texts = utterances["text"]
        if len(texts) == 0:
            continue

        audio_utterances = []
        for idx, text in enumerate(texts):
            utt_id = utterances["utt_id"][idx]
            start = utterances["start"][idx]
            end = utterances["end"][idx]

            audio_key = "-".join(utt_id.split("-")[:-3])
            audio_data = audio_files[audio_key]

            start_samples = int(start * sampling_rate)
            end_samples = int(end * sampling_rate)

            if start_samples > len(audio_data):
                start_samples = len(audio_data)

            extracted_audio = audio_data[start_samples:end_samples]
            if len(extracted_audio) == 0:
                continue

            audio_utterances.append(
                {
                    "utt_id": utt_id,
                    "text": text,
                    "array": extracted_audio.unsqueeze(0),
                }
            )

        for audio_utterance in audio_utterances:
            row = {
                "utt_id": audio_utterance["utt_id"],
                "text": audio_utterance["text"],
                "filename": f'{wav_dir}/{audio_utterance["utt_id"]}.wav',
            }

            # Save the audio file
            torchaudio.save(
                row["filename"],
                audio_utterance["array"],
                sampling_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            row["duration"] = librosa.get_duration(path=row["filename"])

            if row["duration"] == 0:
                os.remove(row["filename"])
                print("Skipping...", row["filename"], "duration is zero")
                continue

            # Append an utterance to the final file
            with open(args.to, "a") as f_to:
                jsonl = json.dumps(row)
                f_to.write(jsonl + "\n")
