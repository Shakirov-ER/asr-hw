import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}

    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    spectrograms = []
    result_batch["spectrogram_length"] = torch.tensor(
        [item["spectrogram"].shape[-1] for item in dataset_items]
    )
    max_spec_dim_last = torch.max(result_batch["spectrogram_length"])
    for item in dataset_items:
        spectrogram = item["spectrogram"]
        spectrograms.append(
            F.pad(
                spectrogram,
                (0, max_spec_dim_last - spectrogram.shape[-1]),
                "constant",
                0,
            )
        )

    result_batch["spectrogram"] = torch.cat(spectrograms, dim=0)

    texts = []
    texts_encoded = []
    result_batch["text_encoded_length"] = torch.tensor(
        [item["text_encoded"].shape[-1] for item in dataset_items]
    )
    max_encoded_text_dim_last = torch.max(result_batch["text_encoded_length"])
    for item in dataset_items:
        text = item["text"]
        texts.append(text)
        text_encoded = item["text_encoded"]
        texts_encoded.append(
            F.pad(
                text_encoded,
                (0, max_encoded_text_dim_last - text_encoded.shape[-1]),
                "constant",
                0,
            )
        )

    result_batch["text_encoded"] = torch.cat(texts_encoded, dim=0)
    result_batch["text"] = texts

    return result_batch
