import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

import pandas as pd

logger = logging.getLogger(__name__)

URL_LINKS = ['https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2']


class LJSpeechDataset(BaseDataset):
    def __init__(self, data_dir=None, split=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self.split = split
        self._data_dir = data_dir
        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _load_part(self):
        arch_path = self._data_dir / f"LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS[0], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    def _get_or_load_index(self):
        index_path = self._data_dir / f"lj_index_{self.split}.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        split_dir = self._data_dir / 'LJSpeech-1.1.tar.bz2'
        if not split_dir.exists():
            self._load_part()

        pass_token = '<pass>'
        df_path = self._data_dir / "metadata.csv"
        df = pd.read_csv(df_path, sep='|', header=None)
        df.fillna(pass_token, inplace=True)
        df = df[:11000] if self.split == "train" else df[11000:]
        wav_dir = self._data_dir / 'wavs'
        for wav_id in tqdm(df[0]):
            if df[df[0] == wav_id][2].values[0] == pass_token:
                continue
            wav_text = df[df[0] == wav_id][2].values[0].lower()
            wav_path = wav_dir / f"{wav_id}.wav"
            t_info = torchaudio.info(str(wav_path))
            length = t_info.num_frames / t_info.sample_rate
            index.append(
                {
                    "path": str(wav_path.absolute().resolve()),
                    "text": wav_text.lower(),
                    "audio_len": length,
                }
            )
        return index


if __name__ == "__main__":
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()
    config_parser = ConfigParser.get_default_configs()

    ds = LJSpeechDataset(
        text_encoder=text_encoder, config_parser=config_parser
    )
    item = ds[0]
    print(item)