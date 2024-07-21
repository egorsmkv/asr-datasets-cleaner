import os
import json
import argparse

from datasets import load_dataset

os.environ["HF_DATASETS_OFFLINE"] = "true"

parser = argparse.ArgumentParser(description="Extract utterances")

parser.add_argument("-dd", "--dataset_dir", required=True)
parser.add_argument("-ss", "--subset", required=True)
parser.add_argument("-cd", "--cache_dir", required=True)
parser.add_argument("-bs", "--batch_size", type=int, required=True)

args = parser.parse_args()

subset = args.subset
data_dir = f"data/{subset}"

ds = load_dataset(
    args.dataset_dir,
    subset,
    data_dir=data_dir,
    trust_remote_code=True,
    cache_dir=args.cache_dir,
)

train_set = ds["train"]
train_set = train_set.remove_columns(["audio"])

ds_iter = train_set.iter(batch_size=args.batch_size)

for batch in ds_iter:
    for idx, utterances in enumerate(batch["utterances"]):
        text = utterances["text"]
        if len(text) == 0:
            continue

        ds_id = batch["id"][idx]
        video_id = batch["video_id"][idx]
        duration = batch["duration"][idx]

        data = json.dumps(
            {
                "id": ds_id,
                "video_id": video_id,
                "duration": duration,
                "utterances": utterances,
            }
        )
        print(data)
