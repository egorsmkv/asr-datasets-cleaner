import json
import argparse

from lingua import LanguageDetectorBuilder

parser = argparse.ArgumentParser(description="Text LID on utterances")

parser.add_argument("-f", "--file", required=True)
parser.add_argument("-t", "--to", required=True)

args = parser.parse_args()

text_lid = (
    LanguageDetectorBuilder.from_all_languages()
    .with_preloaded_language_models()
    .build()
)


jsonlines = []
with open(args.file, "r") as f:
    for line in f:
        jsonlines.append(json.loads(line))


with open(args.to, "w") as f_to:
    for jsonline in jsonlines:
        utterances = jsonline["utterances"]["text"]

        text_lid_scores = []

        for utterance in utterances:
            confidence_values = text_lid.compute_language_confidence_values(utterance)
            confidence_values = [
                {"lang": it.language.iso_code_639_1.name.lower(), "score": it.value}
                for it in confidence_values
                if it.value > 0
            ]

            text_lid_scores.append(confidence_values)

        jsonline["utterances"]["text_normalized"] = text_lid_scores

        f_to.write(json.dumps(jsonline) + "\n")
