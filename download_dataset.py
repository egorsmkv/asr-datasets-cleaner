from datasets import load_dataset

cache_dir = "./cache-yodas2-uk/"

ds = load_dataset("espnet/yodas2", "uk000", cache_dir=cache_dir)

print("Finished")
