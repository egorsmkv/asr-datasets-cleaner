import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser(description="Audio LID on utterances")

parser.add_argument("-bs", "--batch_size", type=int, required=True)
parser.add_argument("-cd", "--cache_dir", required=True)
parser.add_argument("-s", "--subset", required=True)

args = parser.parse_args()

ds = load_dataset("espnet/yodas2", args.subset, cache_dir=args.cache_dir)

train_set = ds["train"]
train_set = train_set.remove_columns(["audio"])

ds_iter = train_set.iter(batch_size=args.batch_size)

total_duration = 0
total_speech_duration = 0

for batch in ds_iter:
    for duration in batch["duration"]:
        total_duration += duration

    for utterances in batch["utterances"]:
        text = utterances["text"]
        if len(text) == 0:
            continue

        for idx, _ in enumerate(text):
            start = utterances["start"][idx]
            end = utterances["end"][idx]

            total_speech_duration += end - start

    print("total duration:", round(total_duration / 60 / 60, 2), "hours")
    print("total speech duration:", round(total_speech_duration / 60 / 60, 2), "hours")

print("---")

print("Total duration:", round(total_duration / 60 / 60, 2), "hours")
print("Total speech duration:", round(total_speech_duration / 60 / 60, 2), "hours")
