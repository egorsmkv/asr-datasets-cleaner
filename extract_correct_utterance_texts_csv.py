import json
import argparse

parser = argparse.ArgumentParser(description="Extract correct utterance texts to CSV")

parser.add_argument("-wd", "--wav_dir", required=True)
parser.add_argument("-f", "--file", required=True)
parser.add_argument("-t", "--to", required=True)

args = parser.parse_args()

jsonlines = []
with open(args.file, "r") as f:
    for line in f:
        jsonlines.append(json.loads(line))

with open(args.to, "w") as f_to:
    for jsonline in jsonlines:
        texts = jsonline["utterances"]["text"]
        utt_ids = jsonline["utterances"]["utt_id"]

        train_set = []
        for idx, text in enumerate(texts):
            if len(text) == 0:
                continue

            utt_id = utt_ids[idx]
            wav_file = f'{args.wav_dir}/{utt_id}.wav'
            
            train_set.append({
                'wav_file': wav_file,
                'text': text,
            })

        for item in train_set:
            f_to.write(f'{item["wav_file"]},{item["text"]}\n')
