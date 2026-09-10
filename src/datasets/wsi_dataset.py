import json
import copy
from pathlib import Path
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import datasets

from .base_dataset import BaseDataset


class WSIDataset(BaseDataset):
    def load_dataset(self):
        with open(self.labels_path, "r") as f:
            self.dataset = json.load(f)

    def _get_sample_at_idx(self, idx: int) -> List[dict]:
        sample = self.dataset[idx]
        filename = f"{sample['filename']}.pt"
        data_dir = Path(self.data_dir)

        conv = copy.deepcopy(sample["conversations"]) # Need to copy if persistent workers are used to avoid modifying the original data
        conv[0]["content"] = [
            {"type" : "text", "text" : conv[0]["content"]},
            {"type" : "features", "features" : torch.load(data_dir / "features" / filename, weights_only=True)},
            {"type" : "coords", "coords" : torch.load(data_dir / "coords" / filename, weights_only=True)}
        ]
        
        return conv

    def __len__(self):
        return len(self.dataset)

    def _compute_lengths(self, batch_size: int, num_proc: int) -> List[int]:
        hf_dataset = datasets.Dataset.from_list(self.dataset)
        tokenizer = self.preprocessor.tokenizer
        token_compression_rate = self.token_compression_rate
        data_dir = Path(self.data_dir)

        def compute_length(batch):
            convs = batch["conversations"]
            filenames = batch["filename"]
            
            token_ids_batch = tokenizer.apply_chat_template(convs, add_generation_prompt=False, tokenize=True, return_dict=False)
            
            lengths = []
            for token_ids, filename in zip(token_ids_batch, filenames):
                coords_path = data_dir / "coords" / f"{filename}.pt"
                num_coords = torch.load(coords_path, weights_only=True).shape[0]
                slide_length = int(num_coords * (1 - token_compression_rate))
                
                conv_length = len(token_ids)
                lengths.append(slide_length + conv_length)
            return {"length": lengths}

        lengths_dataset = hf_dataset.map(
            compute_length,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=hf_dataset.column_names,
            desc="Building lengths"
        )
        
        return list(lengths_dataset["length"])
