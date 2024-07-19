# `asr-datasets-cleaner`

> [!WARNING]  
> Currently, this work is in progress.

> This repository contains a pipeline for better ASR training solving these two tasks: **(1)** remove incorrect audio samples from ASR datasets by LID filtering and **(2)** normalize text samples.

Authors:

- Yehor Smoliakov: [@egorsmkv][4] on GitHub, and <egorsmkv@gmail.com> for private discussions.

## Idea

1. Use https://huggingface.co/facebook/mms-lid-126 to detect the language in audio samples.

2. Use https://huggingface.co/skypro1111/mbart-large-50-verbalization to do text normalization of text samples
(convert numerals to their text form, that is, $5 -> five dollars).

## Details

- We use the *Ukrainian* subset of [YODAS2][1] in our experiment.
- You need to have [uv][2], [nq][3], Python 3.12, and a CUDA device to run the code.

## Install

```shell
uv venv --python 3.12

source .venv/bin/activate

uv pip install -r requirements.txt

# in development mode

uv pip install ruff

ruff check
ruff format
```

## Usage

1. Download the dataset to the local disk:

```shell
nq python download_dataset.py
```

2. Extract utterances:

```shell
python extract_utterances.py --cache_dir cache-yodas2-uk --subset uk000 --batch_size 32 > data/uk000.jsonl

python extract_utterances.py --cache_dir cache2-yodas2-uk --subset uk100 --batch_size 32 > data/uk100.jsonl
```

3. Normalize utterances:

```shell
python normalize_utterances.py --file data/uk000.jsonl --batch_size 8 --device cuda:0 --to data/uk000_normalized.jsonl

python normalize_utterances.py --file data/uk100.jsonl --batch_size 8 --device cuda:1 --to data/uk100_normalized.jsonl
```

2. (optional) Inference audio samples from YODAS2 by difference variants of MMS LID model to see their outputs:

```shell
python mms_lid.py --model_id facebook/mms-lid-126 --cache_dir cache-yodas2-uk --subset uk000 --device cuda:0 > mms-checkpoints-test/mms-lid-126.txt

python mms_lid.py --model_id facebook/mms-lid-256 --cache_dir cache2-yodas2-uk --subset uk100 --device cuda:0 > mms-checkpoints-test/mms-lid-126.txt
```

3. (optional) Inference a text sample by MBART model for text normalization:

```shell
python text_normalization.py
```

## Misc

MMS has these models for the LID task:

- https://huggingface.co/facebook/mms-lid-4017
- https://huggingface.co/facebook/mms-lid-2048
- https://huggingface.co/facebook/mms-lid-1024
- https://huggingface.co/facebook/mms-lid-512
- https://huggingface.co/facebook/mms-lid-256
- https://huggingface.co/facebook/mms-lid-126

[1]: https://huggingface.co/datasets/espnet/yodas2
[2]: https://github.com/astral-sh/uv
[3]: https://github.com/leahneukirchen/nq
[4]: https://github.com/egorsmkv
