import argparse
from os.path import exists

parser = argparse.ArgumentParser(description="Generate commands to run using bash")

parser.add_argument("-dd", "--dataset_dir", required=True)
parser.add_argument("-ss", "--subset", required=True)
parser.add_argument("-mf", "--max_files", type=int, required=True)

args = parser.parse_args()

if not exists(args.dataset_dir):
    print(f"Directory {args.dataset_dir} does not exist")
    exit(1)

dataset_dir = args.dataset_dir
subset = args.subset
max_files = args.max_files
hf_repo = "https://huggingface.co/datasets/espnet/yodas2/resolve/main/data"

# Generate the list of audio files:

audio_files = []
for i in range(max_files + 1):
    filename = str(i).zfill(8)

    save_as = f"{dataset_dir}/data/{subset}/audio/{filename}.tar.gz"
    url = f"{hf_repo}/{subset}/audio/{filename}.tar.gz?download=true"

    audio_files.append(
        {
            "save_as": save_as,
            "url": url,
        }
    )

# Generate the list of duration files:

duration_files = []
for i in range(max_files + 1):
    filename = str(i).zfill(8)

    save_as = f"{dataset_dir}/data/{subset}/duration/{filename}.txt"
    url = f"{hf_repo}/{subset}/duration/{filename}.txt"

    duration_files.append(
        {
            "save_as": save_as,
            "url": url,
        }
    )

# Generate the list of text files:

text_files = []
for i in range(max_files + 1):
    filename = str(i).zfill(8)

    save_as = f"{dataset_dir}/data/{subset}/text/{filename}.json"
    url = f"{hf_repo}/{subset}/text/{filename}.json"

    text_files.append(
        {
            "save_as": save_as,
            "url": url,
        }
    )


# Generate the bash script:

print("#!/bin/bash")
print()

for file in audio_files:
    if not exists(file["save_as"]):
        command = f'nq wget -O {file["save_as"]} {file["url"]}'
        print(command)

print()

for file in duration_files:
    if not exists(file["save_as"]):
        command = f'nq wget -O {file["save_as"]} {file["url"]}'
        print(command)

print()

for file in text_files:
    if not exists(file["save_as"]):
        command = f'nq wget -O {file["save_as"]} {file["url"]}'
        print(command)
