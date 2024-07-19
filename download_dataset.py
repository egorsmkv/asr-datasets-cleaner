from datasets import load_dataset

ds1 = load_dataset("espnet/yodas2", "uk000", cache_dir="./cache-yodas2-uk/")

print("First dataset is downloaded")

ds2 = load_dataset("espnet/yodas2", "uk100", cache_dir= "./cache2-yodas2-uk/")

print("Second dataset is downloaded")
