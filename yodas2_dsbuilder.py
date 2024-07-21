from collections import OrderedDict
from pathlib import Path
import datasets
import json

lang2shard_cnt = {
    "aa000": 2,
    "ab000": 2,
    "af000": 2,
    "ak000": 2,
    "am000": 9,
    "ar000": 154,
    "as000": 2,
    "ay000": 2,
    "az000": 4,
    "ba000": 2,
    "be000": 7,
    "bg000": 12,
    "bh000": 2,
    "bi000": 2,
    "bm000": 2,
    "bn000": 92,
    "bo000": 2,
    "br000": 2,
    "bs000": 2,
    "ca000": 10,
    "co000": 2,
    "cr000": 2,
    "cs000": 24,
    "cy000": 2,
    "da000": 6,
    "de000": 369,
    "de100": 500,
    "de101": 500,
    "de102": 114,
    "dz000": 2,
    "ee000": 2,
    "el000": 18,
    "en000": 500,
    "en001": 500,
    "en002": 500,
    "en003": 500,
    "en004": 500,
    "en005": 500,
    "en006": 500,
    "en007": 437,
    "en100": 500,
    "en101": 500,
    "en102": 500,
    "en103": 500,
    "en104": 500,
    "en105": 500,
    "en106": 500,
    "en107": 500,
    "en108": 500,
    "en109": 500,
    "en110": 500,
    "en111": 500,
    "en112": 500,
    "en113": 500,
    "en114": 500,
    "en115": 500,
    "en116": 500,
    "en117": 500,
    "en118": 500,
    "en119": 500,
    "en120": 500,
    "en121": 500,
    "en122": 500,
    "en123": 500,
    "en124": 500,
    "en125": 500,
    "en126": 500,
    "en127": 500,
    "en128": 500,
    "en129": 62,
    "eo000": 4,
    "es000": 483,
    "es100": 500,
    "es101": 500,
    "es102": 500,
    "es103": 500,
    "es104": 500,
    "es105": 500,
    "es106": 500,
    "es107": 500,
    "es108": 201,
    "et000": 2,
    "eu000": 4,
    "fa000": 12,
    "ff000": 2,
    "fi000": 28,
    "fj000": 2,
    "fo000": 2,
    "fr000": 315,
    "fr100": 500,
    "fr101": 500,
    "fr102": 500,
    "fr103": 401,
    "fy000": 1,
    "ga000": 2,
    "gd000": 2,
    "gl000": 3,
    "gn000": 2,
    "gu000": 8,
    "ha000": 4,
    "hi000": 182,
    "hi100": 7,
    "ho000": 2,
    "hr000": 5,
    "ht000": 3,
    "hu000": 32,
    "hy000": 3,
    "ia000": 2,
    "id000": 493,
    "id100": 500,
    "id101": 419,
    "ie000": 2,
    "ig000": 2,
    "ik000": 2,
    "is000": 2,
    "it000": 185,
    "it100": 500,
    "it101": 432,
    "iu000": 2,
    "iw000": 21,
    "ja000": 211,
    "ja100": 303,
    "jv000": 2,
    "ka000": 4,
    "ki000": 1,
    "kk000": 6,
    "kl000": 2,
    "km000": 10,
    "kn000": 7,
    "ko000": 391,
    "ko100": 500,
    "ko101": 500,
    "ko102": 500,
    "ko103": 287,
    "ks000": 2,
    "ku000": 2,
    "ky000": 4,
    "la000": 2,
    "lb000": 2,
    "lg000": 2,
    "ln000": 2,
    "lo000": 2,
    "lt000": 4,
    "lv000": 2,
    "mg000": 2,
    "mi000": 2,
    "mk000": 2,
    "ml000": 12,
    "mn000": 2,
    "mr000": 18,
    "ms000": 8,
    "mt000": 0,
    "my000": 2,
    "na000": 2,
    "nd000": 1,
    "ne000": 6,
    "nl000": 52,
    "nl100": 263,
    "no000": 17,
    "nv000": 2,
    "oc000": 2,
    "om000": 2,
    "or000": 3,
    "pa000": 5,
    "pl000": 140,
    "ps000": 2,
    "pt000": 202,
    "pt100": 500,
    "pt101": 500,
    "pt102": 500,
    "pt103": 382,
    "qu000": 2,
    "rm000": 2,
    "rn000": 2,
    "ro000": 18,
    "ru000": 500,
    "ru001": 287,
    "ru100": 500,
    "ru101": 500,
    "ru102": 500,
    "ru103": 500,
    "ru104": 500,
    "ru105": 500,
    "ru106": 439,
    "rw000": 2,
    "sa000": 2,
    "sc000": 2,
    "sd000": 2,
    "sg000": 1,
    "sh000": 1,
    "si000": 8,
    "sk000": 6,
    "sl000": 4,
    "sm000": 2,
    "sn000": 2,
    "so000": 4,
    "sq000": 2,
    "sr000": 4,
    "st000": 2,
    "su000": 2,
    "sv000": 17,
    "sw000": 4,
    "ta000": 40,
    "te000": 14,
    "tg000": 2,
    "th000": 113,
    "th100": 2,
    "ti000": 2,
    "tk000": 2,
    "tn000": 2,
    "to000": 2,
    "tr000": 155,
    "tr100": 440,
    "ts000": 1,
    "tt000": 2,
    "ug000": 2,
    "uk000": 63,
    "uk100": 56,
    "ur000": 35,
    "uz000": 8,
    "ve000": 2,
    "vi000": 465,
    "vi100": 500,
    "vi101": 472,
    "vo000": 2,
    "wo000": 2,
    "xh000": 2,
    "yi000": 2,
    "yo000": 2,
    "zh000": 42,
    "zu000": 2,
}


class Yodas2Config(datasets.BuilderConfig):
    """BuilderConfig for Yodas2."""

    def __init__(self, lang, version, **kwargs):
        self.language = lang
        self.base_data_path = f"data/{lang}"

        description = f"Youtube speech to text dataset in {self.language}."
        super(Yodas2Config, self).__init__(
            name=lang,
            version=datasets.Version(version),
            description=description,
            **kwargs,
        )


DEFAULT_CONFIG_NAME = "all"
LANGS = list(lang2shard_cnt.keys())
VERSION = "1.0.0"


class Yodas2(datasets.GeneratorBasedBuilder):
    """YodasSample dataset."""

    BUILDER_CONFIGS = [Yodas2Config(lang, version=VERSION) for lang in LANGS]

    VERSION = datasets.Version("1.0.1")

    def _info(self):
        return datasets.DatasetInfo(
            description="Yodas Sample",
            features=datasets.Features(
                OrderedDict(
                    [
                        ("id", datasets.Value("string")),
                        ("video_id", datasets.Value("string")),
                        ("duration", datasets.Value("float")),
                        ("audio", datasets.Audio(sampling_rate=24_000)),
                        (
                            "utterances",
                            datasets.Sequence(
                                feature={
                                    "utt_id": datasets.Value(dtype="string"),
                                    "text": datasets.Value(dtype="string"),
                                    "start": datasets.Value(dtype="float"),
                                    "end": datasets.Value(dtype="float"),
                                }
                            ),
                        ),
                    ]
                )
            ),
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        print(self.config)
        if self.config.data_files is None:
            audio_path = Path(self.base_path) / self.config.data_dir / "audio"
            total_cnt = len(list(audio_path.glob("*.tar.gz")))
        else:
            total_cnt = len(list(self.config.data_files["train"])) // 3

        idx_lst = [f"{i:08d}" for i in range(total_cnt)]
        audio_tar_files = dl_manager.download(
            [f"{self.config.data_dir}/audio/{i:08d}.tar.gz" for i in range(total_cnt)]
        )
        text_files = dl_manager.download(
            [f"{self.config.data_dir}/text/{i:08d}.json" for i in range(total_cnt)]
        )
        duration_files = dl_manager.download(
            [f"{self.config.data_dir}/duration/{i:08d}.txt" for i in range(total_cnt)]
        )

        if dl_manager.is_streaming:
            audio_archives = [
                dl_manager.iter_archive(audio_tar_file)
                for audio_tar_file in audio_tar_files
            ]
            text_archives = [dl_manager.extract(text_file) for text_file in text_files]
            duration_archives = [
                dl_manager.extract(duration_file) for duration_file in duration_files
            ]

        else:
            print("extracting audio ...")
            print("audio tarfiles... ")
            print(audio_tar_files)
            extracted_audio_archives = dl_manager.extract(audio_tar_files)
            print("extracted archives...")
            print(extracted_audio_archives)

            audio_archives = []
            text_archives = []
            duration_archives = []
            for idx, audio_tar_file, extracted_dir, text_file, duration_file in zip(
                idx_lst,
                audio_tar_files,
                extracted_audio_archives,
                text_files,
                duration_files,
            ):
                audio_archives.append(extracted_dir)
                text_archives.append(text_file)
                duration_archives.append(duration_file)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "is_streaming": dl_manager.is_streaming,
                    "audio_archives": audio_archives,
                    "text_archives": text_archives,
                    "duration_archives": duration_archives,
                },
            ),
        ]

    def _generate_examples(
        self, is_streaming, audio_archives, text_archives, duration_archives
    ):
        """Yields examples."""

        global_id = 0

        if is_streaming:
            for tar_file, text_file, duration_file in zip(
                audio_archives, text_archives, duration_archives
            ):
                # video to text
                video2text = {}

                json_obj_lst = json.loads(open(text_file, "r").read())
                for json_obj in json_obj_lst:
                    video_id = json_obj["audio_id"]
                    video2text[video_id] = []

                    for k, v in sorted(json_obj["text"].items()):
                        fields = k.split("-")
                        start_timestamp = float(fields[-2]) / 100
                        end_timestamp = float(fields[-1]) / 100
                        video2text[video_id].append(
                            {
                                "utt_id": k,
                                "text": v,
                                "start": start_timestamp,
                                "end": end_timestamp,
                            }
                        )

                # video to duration
                video2duration = {}
                with open(duration_file) as f:
                    for id_, row in enumerate(f):
                        fields = row.strip().split()
                        video_id = fields[0]
                        duration = float(fields[1])
                        video2duration[video_id] = duration

                for path, audio_f in tar_file:
                    path = Path(path)
                    video_id = path.stem

                    if video_id in video2text and video_id in video2duration:
                        result = {
                            "id": global_id,
                            "video_id": video_id,
                            "audio": {"path": None, "bytes": audio_f.read()},
                            "duration": video2duration[video_id],
                            "utterances": video2text[video_id],
                        }

                        yield global_id, result
                        global_id += 1
        else:
            for extracted_dir, text_file, duration_file in zip(
                audio_archives, text_archives, duration_archives
            ):
                print("extracted_dir ", extracted_dir)

                print("actual extracted dir", extracted_dir)

                # video to text
                video2text = {}
                json_obj_lst = json.loads(open(text_file, "r").read())
                for json_obj in json_obj_lst:
                    video_id = json_obj["audio_id"]
                    video2text[video_id] = []

                    for k, v in sorted(json_obj["text"].items()):
                        fields = k.split("-")
                        start_timestamp = float(fields[-2]) / 100
                        end_timestamp = float(fields[-1]) / 100
                        video2text[video_id].append(
                            {
                                "utt_id": k,
                                "text": v,
                                "start": start_timestamp,
                                "end": end_timestamp,
                            }
                        )

                # video to duration
                video2duration = {}
                with open(duration_file) as f:
                    for id_, row in enumerate(f):
                        fields = row.strip().split()
                        video_id = fields[0]
                        duration = float(fields[1])
                        video2duration[video_id] = duration

                for audio_file in list(Path(extracted_dir).glob("*")):
                    video_id = audio_file.stem

                    if video_id in video2text and video_id in video2duration:
                        result = {
                            "id": global_id,
                            "video_id": video_id,
                            "duration": video2duration[video_id],
                            "audio": {
                                "path": str(audio_file.absolute()),
                                "bytes": open(audio_file, "rb").read(),
                            },
                            "utterances": video2text[video_id],
                        }

                        yield global_id, result
                        global_id += 1
