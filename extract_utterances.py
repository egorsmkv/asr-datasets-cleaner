import json
import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser(description="Extract utterances")

parser.add_argument("-cd", "--cache_dir", required=True)
parser.add_argument("-s", "--subset", required=True)

args = parser.parse_args()

ds = load_dataset("espnet/yodas2", args.subset, cache_dir=args.cache_dir)

batch_size = 32

train_set = ds["train"]
train_set = train_set.remove_columns(["audio"])

ds_iter = train_set.iter(batch_size=batch_size)

for i, batch in enumerate(ds_iter):
    for ii in range(len(batch["utterances"])):
        utterances = batch["utterances"][ii]
        ds_id = batch["id"][ii]
        video_id = batch["video_id"][ii]
        duration = batch["duration"][ii]

        data = json.dumps(
            {
                "id": ds_id,
                "video_id": video_id,
                "duration": duration,
                "utterances": utterances,
            }
        )
        print(data)
