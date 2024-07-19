import argparse

from lingua import LanguageDetectorBuilder
from datasets import load_dataset

parser = argparse.ArgumentParser(description="lingua-py LID")

parser.add_argument("-cd", "--cache_dir", required=True)
parser.add_argument("-s", "--subset", required=True)

args = parser.parse_args()

ds = load_dataset("espnet/yodas2", args.subset, cache_dir=args.cache_dir)

text_lid = (
    LanguageDetectorBuilder.from_all_languages()
    .with_preloaded_language_models()
    .build()
)

n_samples = 10

train_set = ds["train"]
train_set = train_set.remove_columns(["audio"])
train_iterator = iter(train_set)

for _ in range(n_samples):
    sample = next(train_iterator)

    ids = sample["utterances"]["utt_id"]
    text = sample["utterances"]["text"]
    start = sample["utterances"]["start"]
    end = sample["utterances"]["end"]

    for idx, _ in enumerate(text):
        start_a = start[idx]
        end_a = end[idx]
        text_a = text[idx]

        print(end_a - start_a, "||", text_a)

        confidence_values = text_lid.compute_language_confidence_values(text_a)
        confidence_values = [
            {"lang": it.language.iso_code_639_1.name.lower(), "score": it.value}
            for it in confidence_values
            if it.value > 0
        ]

        print(confidence_values)

    print("----" * 5)
