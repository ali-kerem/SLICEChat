from __future__ import annotations
from typing import Sequence, Iterator, Dict, Any

import math, random
import torch
from torch.utils.data import Sampler


def _ddp_info() -> tuple[int, int]:
    """Return (rank, world_size) whether DDP is initialized or not."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _build_global_order(
    lengths: Sequence[int],
    world_size: int,
    per_device_batch_size: int,
    mega_batch_mult: int,
    torch_gen: torch.Generator,
    py_rand: random.Random,
    shuffle: bool,
) -> list[int]:
    """
    1) Random permute globally (torch for DDP seeding control).
    2) Partition into mega-batches of size: mega_batch_mult * (per_device_batch_size * world_size).
    3) Inside each mega-batch: sort by length desc, chunk into groups of size `global_batch`,
       shuffle the group order, then flatten.
       This keeps each step's ranks working on similarly-long samples with less padding.
    """
    n = len(lengths)
    if n == 0:
        return []

    global_batch = per_device_batch_size * world_size
    mega_batch_size = max(global_batch, mega_batch_mult * global_batch)

    if shuffle:
        perm = torch.randperm(n, generator=torch_gen).tolist()
    else:
        perm = list(range(n))

    out: list[int] = []

    for start in range(0, n, mega_batch_size):
        block = perm[start:start + mega_batch_size]
        # sort by length desc
        block.sort(key=lambda i: lengths[i], reverse=True)
        
        # Group by global_batch (all ranks' batches combined) to preserve local length similarity
        # This ensures that when we stride later (rank::world_size), all ranks get similar lengths
        # and the variance within a single rank's batch is minimized.
        groups = [block[i:i + global_batch] for i in range(0, len(block), global_batch)]
        
        # shuffle group order to keep randomness across steps, but PRESERVE the sorted internals of the batch
        if shuffle:
            py_rand.shuffle(groups)
        for g in groups:
            out.extend(g)

    return out


class LengthGroupedDistributedSampler(Sampler[int]):
    """
    Distributed length-grouped sampler.
    - Groups examples by length within mega-batches.
    - Ensures even divisibility per epoch (drop_last or pad).
    - Returns rank-specific indices (one sampler instance per rank).
    - Reproducible across epochs via set_epoch(epoch).
    """

    def __init__(
        self,
        lengths: Sequence[int],
        per_device_batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        mega_batch_mult: int = 50,  # how many global batches per mega-batch
    ) -> None:
        super().__init__(None)
        self.lengths = lengths
        self.per_device_batch_size = int(per_device_batch_size)
        assert self.per_device_batch_size > 0
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.mega_batch_mult = int(mega_batch_mult)

        self.rank, self.world_size = _ddp_info()
        self.global_batch = self.per_device_batch_size * self.world_size

        # Epoch/step state for resuming mid-epoch
        self.epoch = 0
        self.step_offset = 0  # number of *global* steps already consumed in this epoch

        # Compute epoch sizes following DistributedSampler semantics
        n = len(self.lengths)
        if self.drop_last:
            self.total_size = (n // self.global_batch) * self.global_batch
        else:
            self.total_size = math.ceil(n / self.global_batch) * self.global_batch
        self.num_samples = self.total_size // self.world_size

        # RNGs
        self._torch_gen = torch.Generator()
        self._py_rand = random.Random()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step_offset": self.step_offset,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.step_offset = int(state.get("step_offset", 0)) % max(self.num_samples, 1)

    def __iter__(self) -> Iterator[int]:
        # Seed RNGs per epoch
        base = self.seed + self.epoch
        self._torch_gen.manual_seed(base)
        self._py_rand.seed(base)

        order = _build_global_order(
            self.lengths,
            world_size=self.world_size,
            per_device_batch_size=self.per_device_batch_size,
            mega_batch_mult=self.mega_batch_mult,
            torch_gen=self._torch_gen,
            py_rand=self._py_rand,
            shuffle=self.shuffle,
        )

        # Pad or trim to total_size
        if len(order) < self.total_size:
            # repeat from the start to pad
            order = (order * ((self.total_size // max(len(order), 1)) + 1))[: self.total_size]
        elif len(order) > self.total_size:
            # trim if drop_last
            order = order[: self.total_size]

        # Convert global interleaved order -> rank-specific slice.
        # After length-grouping, each "group" has size global_batch, so this stride slicing
        # maps naturally to ranks.
        per_rank = order[self.rank:self.total_size:self.world_size]

        # Skip already-consumed samples on this rank for resuming mid-epoch
        if self.step_offset:
            skip = self.step_offset * self.per_device_batch_size
            per_rank = per_rank[skip:]

        yield from per_rank
        # reset offset after an iteration
        self.step_offset = 0
