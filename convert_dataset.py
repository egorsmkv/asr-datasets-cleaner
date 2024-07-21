import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Convert the dataset to `datasets` format")

parser.add_argument("-dd", "--dataset_dir", required=True)
parser.add_argument("-ss", "--subset", required=True)
parser.add_argument("-mf", "--max_files", type=int, required=True)
parser.add_argument("-cd", "--cache_dir", required=True)

args = parser.parse_args()

dataset_dir = args.dataset_dir
subset = args.subset
data_dir = f"{dataset_dir}/data/{subset}"
max_files = args.max_files
cache_dir = args.cache_dir


train_files = []
for i in range(max_files + 1):
    filename = str(i).zfill(8)

    tar_file = f"data/{subset}/audio/{filename}.tar.gz"
    duration_file = f"data/{subset}/duration/{filename}.txt"
    json_file = f"data/{subset}/text/{filename}.json"

    train_files.append(tar_file)
    train_files.append(duration_file)
    train_files.append(json_file)


ds = load_dataset(
    data_dir,
    data_files={
        "train": train_files,
    },
    trust_remote_code=True,
    data_dir=data_dir,
    cache_dir=cache_dir,
)

print("Converted")
