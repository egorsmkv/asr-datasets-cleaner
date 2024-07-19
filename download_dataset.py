from datasets import load_dataset

cache_dir = "./cache2-yodas2-uk/"

ds = load_dataset("espnet/yodas2", "uk100", cache_dir=cache_dir)

print("Finished")
