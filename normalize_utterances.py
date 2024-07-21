import json
import argparse

from transformers import MBartForConditionalGeneration, AutoTokenizer

parser = argparse.ArgumentParser(description="Normalize utterances")

parser.add_argument("-f", "--file", required=True)
parser.add_argument("-t", "--to", required=True)
parser.add_argument("-d", "--device", required=True)
parser.add_argument("-bs", "--batch_size", type=int, required=True)

args = parser.parse_args()

model_name = "skypro1111/mbart-large-50-verbalization"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.src_lang = "uk_XX"
tokenizer.tgt_lang = "uk_XX"

model = MBartForConditionalGeneration.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    device_map=args.device,
)
model.eval()

jsonlines = []
with open(args.file, "r") as f:
    for line in f:
        jsonlines.append(json.loads(line))


def make_batches(iterable, n=1):
    lx = len(iterable)
    for ndx in range(0, lx, n):
        yield iterable[ndx : min(ndx + n, lx)]


with open(args.to, "w") as f_to:
    for jsonline in jsonlines:
        utterances = jsonline["utterances"]["text"]

        text_normalized = []
        input_texts = ["<verbalization>:" + utt for utt in utterances]

        for batch in make_batches(input_texts, args.batch_size):
            encoded_input = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(args.device)

            output_ids = model.generate(
                **encoded_input, max_length=1024, num_beams=5, early_stopping=True
            )
            normalized_utterances = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )

            text_normalized.extend(normalized_utterances)

        jsonline["utterances"]["text_normalized"] = text_normalized

        f_to.write(json.dumps(jsonline) + "\n")
