import json
import argparse

parser = argparse.ArgumentParser(description="Filter utterance texts by language")

parser.add_argument("-f", "--file", required=True)
parser.add_argument("-t", "--to", required=True)
parser.add_argument("-l", "--language", required=True)
parser.add_argument("-s", "--score", type=float, required=True)

args = parser.parse_args()

correct_utterance_texts = 0
incorrect_utterance_texts = 0

jsonlines = []
with open(args.file, "r") as f:
    for line in f:
        jsonlines.append(json.loads(line))

with open(args.to, "w") as f_to:
    for jsonline in jsonlines:
        texts = jsonline["utterances"]["text"]

        replaced_texts = []

        for idx, text in enumerate(texts):
            scores = jsonline["utterances"]["text_lid_scores"][idx]
            if len(scores) == 0:
                continue

            row = max(scores, key=lambda x: x["score"])
            if row["lang"] != args.language:
                incorrect_utterance_texts += 1
                replaced_texts.append("")
                continue
            if row["score"] < args.score:
                incorrect_utterance_texts += 1
                replaced_texts.append("")
                continue

            correct_utterance_texts += 1
            replaced_texts.append(text)

        jsonline["utterances"]["text"] = replaced_texts
        del jsonline["utterances"]["text_lid_scores"]

        f_to.write(json.dumps(jsonline) + "\n")

print("---")

print(f"Correct {correct_utterance_texts} utterance texts")
print(f"Incorrect {incorrect_utterance_texts} utterance texts")
