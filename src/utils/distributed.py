import os
import torch


def get_ranks():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, local_rank


def get_num_nodes():
    return int(os.environ.get('SLURM_NNODES', '1'))


def calculate_accumulation_steps(config):
    num_nodes = get_num_nodes()

    # Calculate number of GPUs per node
    devices = config["train_args"]["devices"]
    if devices == "auto":
        gpus_per_node = torch.cuda.device_count()
    elif isinstance(devices, int):
        gpus_per_node = devices
    elif isinstance(devices, list):
        gpus_per_node = len(devices)
    else:
        raise ValueError(f"Invalid devices configuration: {devices}")

    total_gpus = num_nodes * gpus_per_node
    global_batch_size = config["train_args"]["global_batch_size"]
    batch_size = config["data"]["dataloader_args"]["batch_size"]
    
    accumulation_steps = global_batch_size // (batch_size * total_gpus)

    assert accumulation_steps > 0, f"Accumulation steps must be greater than 0, but got {accumulation_steps}"

    rank, local_rank = get_ranks()
    if rank == 0 and local_rank == 0:
        print(f"Number of nodes: {num_nodes}", flush=True)
        print(f"Number of GPUs per node: {gpus_per_node}", flush=True)
        print(f"Total number of GPUs: {total_gpus}", flush=True)
        print(f"Calculated accumulation steps: {accumulation_steps}", flush=True)
        print(f"Global batch size: {global_batch_size}", flush=True)
        print(f"Per-device batch size: {batch_size}", flush=True)

    return accumulation_steps
