# `asr-datasets-cleaner`

> [!WARNING]  
> Currently, this work is in progress.

> This repository contains a pipeline for better ASR training solving these two tasks: **(1)** remove incorrect audio samples from ASR datasets by LID filtering and **(2)** normalize text samples.

Authors:

- Yehor Smoliakov: [@egorsmkv][4] on GitHub, and <egorsmkv@gmail.com> for private discussions.

## Idea

1. Use https://huggingface.co/facebook/mms-lid-126 to detect the language in audio samples.

2. Use https://github.com/pemistahl/lingua-py to detect the language in text samples.

3. Use https://huggingface.co/skypro1111/mbart-large-50-verbalization to do text normalization 
(convert numerals/abbreviations to their textual representation, that is, $5 -> five dollars).

## Details

- We use the *Ukrainian* subset of [YODAS2][1] in our command examples.
- We patch the YODAS2's dataset builder script to download only a part of the dataset.

## Required software

- Python 3.12
- [uv][2]
- [nq][3]
- CUDA device

## Install

```shell
uv venv --python 3.12

source .venv/bin/activate

uv pip install -r requirements.txt

# in development mode
uv pip install -r requirements-dev.txt
```

## Usage

1. Generate a bash file to download required files from [YODAS2][1]:

```shell
python generate_commands.py --dataset_dir `pwd`/uk_yodas2 --subset uk000 --max_files 10 > download_dataset.sh
```

2. Download the dataset:

```shell
bash download_dataset.sh
```

3. Convert the dataset to `datasets` format:

Copy the `yodas2_dsbuilder.py` file to your `dataset_dir` directory and rename it as `dataset_dir`. So in the following example, the `dataset_dir` is `uk_yodas2` and the script must be renamed as `uk_yodas2.py`.

Then convert the dataset, it will unarchive files and generate metadata:

```shell
python convert_dataset.py --dataset_dir `pwd`/uk_yodas2 --subset uk000 --max_files 10 --cache_dir cache-yodas2-uk000
```

4. Extract utterances:

```shell
python extract_utterances.py --dataset_dir `pwd`/uk_yodas2 --subset uk000 --cache_dir ../cache-yodas2-uk000 --batch_size 128 > data/uk000.jsonl
```

5. Text LID:

```shell
python text_lid.py --file data/uk000.jsonl --to data/uk000_+tlid.jsonl
```

6. Filter by a language:

```shell
python filter_by_language.py --file data/uk000_+tlid.jsonl --to data/uk000_+only_uk.jsonl --language uk --score 0.95
```

7. Audio LID:

```shell
python audio_lid.py --dataset_dir `pwd`/uk_yodas2 --subset uk000 --cache_dir ../cache-yodas2-uk000 --batch_size 16 --model_id facebook/mms-lid-126 --file data/uk000_+tlid.jsonl --to data/uk000_+tlid_+alid.jsonl --device cuda:0
```

8. Normalize utterances:

```shell
python normalize_utterances.py --file data/uk000.jsonl --to data/uk000_normalized.jsonl --batch_size 8 --device cuda:0
```

## Examples

0. Go to `examples/`

1. Inference audio samples by the different variants of MMS LID model to see their outputs:

```shell
python audio_lid.py --model_id facebook/mms-lid-126 --dataset_dir `pwd`/../uk_yodas2 --subset uk000 --cache_dir ../cache-yodas2-uk000 --device cuda:0 > ../mms-checkpoints-test/mms-lid-126.txt
```

2. Inference text samples by lingua-py to see their text language:

```shell
python text_lid.py --dataset_dir `pwd`/../uk_yodas2 --subset uk000 --cache_dir ../cache-yodas2-uk000
```

3. Inference text samples by the MBART model for text normalization:

```shell
python normalize_utterances.py
```

4. Calculate the duration of the dataset:

```shell
python count_durations.py --dataset_dir `pwd`/../uk_yodas2 --subset uk000 --cache_dir ../cache-yodas2-uk000 --batch_size 128
```

## Development

```shell
ruff check
ruff format
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
