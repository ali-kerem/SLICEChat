from typing import Literal
import random
import warnings

import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

from src.datasets import DatasetFactory
from src.data_utils.collators import CollatorFactory
from src.data_utils.preprocessors import BasePreprocessor
from src.data_utils.sampler import LengthGroupedDistributedSampler


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MLLMDataModule(L.LightningDataModule):
    def __init__(
        self,
        modality: Literal["slide"],
        dataset_args: dict,
        dataloader_args: dict,
        preprocessor: BasePreprocessor,
    ):
        super().__init__()
        self.modality = modality
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args
        self.preprocessor = preprocessor

        self.data_collator = CollatorFactory.create(modality=self.modality)
        self.sampler = None

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = DatasetFactory.create(preprocessor=self.preprocessor, 
                                                        build_lengths=self.dataloader_args["length_grouped_sampler"],
                                                        **self.dataset_args)
            
            # Create sampler once during setup if using length-grouped sampling
            if self.dataloader_args["length_grouped_sampler"]:
                if not hasattr(self.train_dataset, "lengths"):
                    warnings.warn(f"LengthGroupedDistributedSampler is not supported for {self.dataset_args['dataset_name']} dataset. Using default sampler instead.")
                else:
                    self.sampler = LengthGroupedDistributedSampler(
                        lengths=self.train_dataset.lengths,
                        per_device_batch_size=self.dataloader_args["batch_size"],
                        shuffle=self.dataloader_args["shuffle"],
                        seed=self.dataloader_args["seed"],
                        drop_last=True,
                        mega_batch_mult=50
                    )

    def train_dataloader(self):
        dataloader_args = self.dataloader_args.copy()
        del dataloader_args["length_grouped_sampler"]
        seed = dataloader_args.pop("seed")
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # If using length-grouped sampler, disable DataLoader's shuffle to suppress errors.
        # We are already shuffling the dataset in the sampler.
        if self.sampler is not None:
            dataloader_args["shuffle"] = False

        return DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            sampler=self.sampler,
            generator=generator,
            worker_init_fn=seed_worker,
            persistent_workers=dataloader_args["num_workers"] > 0,
            **dataloader_args
        )
