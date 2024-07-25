import shutil
import argparse
from os.path import exists
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Copy correct WAVs to a destination")

parser.add_argument("-cf", "--correct_files", required=True)
parser.add_argument("-df", "--dst_folder", required=True)

args = parser.parse_args()

if not exists(args.dst_folder):
    print(f"Destination folder {args.dst_folder} does not exist")
    exit(1)

total_lines = 0
with open(args.correct_files) as f:
    for _ in f:
        total_lines += 1

with open(args.correct_files) as f:
    for idx, line in tqdm(enumerate(f), total=total_lines):
        if idx == 0:
            continue
        wav = line.strip()
        parts = wav.split(",")
        if len(parts) != 2:
            print('incorrect format of the line')
            continue

        filename = parts[0].split('/')[-1]
        file_dst = f"{args.dst_folder}/{filename}"

        if exists(file_dst):
            continue

        shutil.copy(parts[0], file_dst)

print('Done')
