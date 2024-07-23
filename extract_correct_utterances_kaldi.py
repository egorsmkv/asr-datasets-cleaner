import json
import argparse

parser = argparse.ArgumentParser(
    description="Extract correct utterances in the Kaldi format"
)

parser.add_argument("-f", "--file", required=True)
parser.add_argument("-wd", "--wav_dir", required=True)
parser.add_argument("-w", "--wav_scp", required=True)
parser.add_argument("-t", "--text", required=True)

args = parser.parse_args()

jsonlines = []
with open(args.file, "r") as f:
    for line in f:
        jsonlines.append(json.loads(line))

with open(args.wav_scp, "w") as f_wav_scp, open(args.text, "w") as f_text:
    for jsonline in jsonlines:
        utt_ids = jsonline["utterances"]["utt_id"]
        texts = jsonline["utterances"]["text"]

        train_set = []
        for idx, text in enumerate(texts):
            if len(text) == 0:
                continue
            utt_id = utt_ids[idx]
            wav_file = f"{args.wav_dir}/{utt_id}.wav"

            train_set.append(
                {
                    "utt_id": utt_id,
                    "wav_file": wav_file,
                    "text": text,
                }
            )

        for item in train_set:
            f_wav_scp.write(f'{item["utt_id"]}\t{item["wav_file"]}' + "\n")
            f_text.write(f'{item["utt_id"]}\t{item["text"]}' + "\n")
