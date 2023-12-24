from torchaudio.models.decoder import download_pretrained_files
from pathlib import Path

files = download_pretrained_files("librispeech-4-gram")
Path(files.lm).rename("language_models/lm.bin")