import torch
from dataclasses import dataclass
from typing import Dict, Sequence

from src.data_utils.collators import BaseCollator


@dataclass
class SlideCollator(BaseCollator):
    def __call__(self, data_dicts: Sequence[Dict]) -> Dict:
        data_dict = super().__call__(data_dicts)

        features = [data_dict["features"] for data_dict in data_dicts]
        coords = [data_dict["coords"] for data_dict in data_dicts]
        data_dict["features"] = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        data_dict["coords"] = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True, padding_value=-1)
        data_dict["padding_mask"] = (data_dict["coords"] == -1).any(dim=-1).long()

        return data_dict
