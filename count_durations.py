import time
from datasets import load_dataset

cache_dir = "./cache-yodas2-uk/"
subset = "uk000"

ds = load_dataset("espnet/yodas2", subset, cache_dir=cache_dir)

test_run = False
batch_size = 32

train_set = ds["train"]
train_set = train_set.remove_columns(["audio"])

ds_iter = train_set.iter(batch_size=batch_size)

total_duration = 0
total_speech_duration = 0

# batch = dict_keys(['id', 'video_id', 'duration', 'audio', 'utterances'])

t0 = time.time()
for i, batch in enumerate(ds_iter):
    for ii in range(len(batch["utterances"])):
        start = batch["utterances"][ii]["start"]
        end = batch["utterances"][ii]["end"]

        duration = batch["duration"][ii]
        speech_duration = sum(
            [end[i] - start_value for i, start_value in enumerate(start)]
        )

        total_duration += duration
        total_speech_duration += speech_duration

    print("total duration:", round(total_duration / 60 / 60, 2), "hours")
    print("total speech duration:", round(total_speech_duration / 60 / 60, 2), "hours")

    if test_run:
        if i == 10:
            break

print("---")

print("Total duration:", round(total_duration / 60 / 60, 2), "hours")
print("Total speech duration:", round(total_speech_duration / 60 / 60, 2), "hours")

print("---")

print("Elapsed time:", time.time() - t0, "seconds")
