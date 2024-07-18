# `asr-datasets-cleaner`

> This repository shows how to make a pipeline that can be used for **(1)** to remove incorrect audio samples from ASR datasets by LID filtering and **(2)** to normalize text samples for better ASR training. We use *Ukrainian* as an example.

Authors:

- Yehor Smoliakov: @egorsmkv on GitHub, and egorsmkv@gmail.com for private discussions.

> Currently, this work is in progress.

## Main idea

1. Use https://huggingface.co/facebook/mms-lid-126 to detect the language in audio samples.

2. Use https://huggingface.co/skypro1111/mbart-large-50-verbalization to do text normalization of text samples
(i.e.: convert numerals to their text form $5 -> five dollars).

## Some details

- We use [YODAS2][1] as a test dataset for our experiment.
- You need to have [uv][2], [nq][3], Python 3.12, and a GPU card to run the code.

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

1. Download the dataset locally:

```shell
nq python download_dataset.py
```

2. (optional) Inference audio samples from YODAS2 by difference variants of MMS LID model to see their outputs:

```shell
python mms_lid_126.py > mms-checkpoints-test/mms-lid-126.txt

python mms_lid_256.py > mms-checkpoints-test/mms-lid-256.txt
```

3. (optional) Inference a text sample by MBART model for text normalization:

```shell
python verbalize_text.py
```

[1]: https://huggingface.co/datasets/espnet/yodas2
[2]: https://github.com/astral-sh/uv
[3]: https://github.com/leahneukirchen/nq
