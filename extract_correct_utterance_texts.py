import json
import argparse

parser = argparse.ArgumentParser(description="Extract correct utterance texts")

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

        correct_texts = []
        for idx, text in enumerate(texts):
            if len(text) == 0:
                continue
            correct_texts.append(text)

        for text in correct_texts:
            f_to.write(text + "\n")
