from dataclasses import dataclass
from typing import Dict, Sequence

@dataclass
class BaseCollator:
    def __call__(self, data_dicts: Sequence[Dict]) -> Dict:
        input_ids = [data_dict["input_ids"] for data_dict in data_dicts]

        data_dict = {
            "input_ids" : input_ids,
        }

        if "labels" in data_dicts[0]:
            labels = [data_dict["labels"] for data_dict in data_dicts]
            data_dict["labels"] = labels

        return data_dict
