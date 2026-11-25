import os

import torch
import torch.distributed as dist


def init_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    elif torch.npu.is_available():
        backend = "hccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return local_rank


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def is_main_process():
    rank, _ = get_dist_info()
    return rank == 0


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensors(use_dist: bool, *tensors: torch.Tensor):
    """All-reduce mixed-shape tensors in one HCCL call.

    Scalars are unsqueezed to (1,) before concatenation and
    squeezed back afterward.

    Args:
        use_dist: whether distributed training is active.
        *tensors: tensors to reduce.  Must all be on the same device.

    Returns:
        A single tensor if one was passed, otherwise a tuple.
    """
    if not use_dist or not tensors:
        return tensors[0] if len(tensors) == 1 else tensors

    device = tensors[0].device
    flat = []
    shapes = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device)
        elif t.device != device:
            t = t.to(device)
        shapes.append(t.shape)
        flat.append(t.unsqueeze(0) if t.ndim == 0 else t)

    cat = torch.cat(flat, dim=0)
    dist.all_reduce(cat)
    cat /= dist.get_world_size()

    results = []
    i = 0
    for shape in shapes:
        n = 1 if shape == torch.Size() else shape[0]
        t = cat[i:i + n]
        if shape == torch.Size():
            t = t.squeeze(0)
        results.append(t)
        i += n

    return results[0] if len(results) == 1 else tuple(results)
