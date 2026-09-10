import os
from typing import List
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from src.data_utils.preprocessors import BasePreprocessor


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 labels_path: str = None,
                 data_dir: str = None,
                 preprocessor: BasePreprocessor = None,
                 token_compression_rate: float = 0.8,
                 system_prompt: str = None,
                 build_lengths: bool = False,
                 return_labels: bool = True):
        """
        Args:
            labels_path: Path to the labels file
            data_dir: Root directory containing features/ and coords/ tensor directories.
            preprocessor: Preprocessor to use for preprocessing the data
            token_compression_rate: Rate of token compression for the slide modality
            system_prompt: System prompt to use for the dataset
            build_lengths: Whether to build the lengths of the dataset, required for length-grouped sampler
            return_labels: Whether to build training labels. Set False for user-only inference conversations.
        """

        super().__init__()
        self.preprocessor = preprocessor
        self.labels_path = labels_path
        self.data_dir = data_dir
        self.token_compression_rate = token_compression_rate
        self.system_prompt = system_prompt
        self.return_labels = return_labels

        self.load_dataset()
        
        if build_lengths:
            self.build_lengths()

    @abstractmethod
    def load_dataset(self):
        """Load dataset-specific data
        Sets self.dataset to a format that can be used by __get_sample_at_idx method"""
        pass

    @abstractmethod
    def _get_sample_at_idx(self, idx: int) -> List[dict]:
        """
        Returns:
            conv: List[dict] (conversation in OpenAI / HF format)
        """
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        conv = self._get_sample_at_idx(idx)

        if self.system_prompt is not None:
            conv.insert(0, {
                "role" : "system",
                "content" : self.system_prompt
            })
        
        data_dict = self.preprocessor(conv, return_labels=self.return_labels)

        return data_dict

    def build_lengths(self, batch_size: int = 2048, num_proc: int = 16):
        """Build sequence lengths for length-grouped sampling.
        
        Only rank 0 computes lengths to avoid memory issues from multiprocess forking.
        Results are broadcast to all other ranks.
        """
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dataset_size = len(self)
        
        if rank == 0:
            self.lengths = self._compute_lengths(batch_size, num_proc)
        
        if world_size > 1 and dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            
            if rank == 0:
                lengths_tensor = torch.tensor(self.lengths, dtype=torch.long, device=device)
            else:
                lengths_tensor = torch.zeros(dataset_size, dtype=torch.long, device=device)
            
            dist.broadcast(lengths_tensor, src=0)
            self.lengths = lengths_tensor.cpu().tolist()

    def _compute_lengths(self, batch_size: int, num_proc: int) -> List[int]:
        """Compute and return the list of sequence lengths. Only called on rank 0.
        
        Subclasses that support length-grouped sampling should override this method.
        
        Args:
            batch_size: Batch size for parallel processing
            num_proc: Number of processes for parallel processing
            
        Returns:
            List of sequence lengths for each sample in the dataset
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _compute_lengths. "
            "Set build_lengths=False or implement _compute_lengths in the subclass."
        )
