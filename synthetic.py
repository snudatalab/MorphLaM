from __future__ import annotations

"""Synthetic tensor generators used by all benchmark scripts."""

from typing import Sequence

import torch

from common import normalize_fro


def _sample(shape: Sequence[int], distribution: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    distribution = str(distribution).lower()
    if distribution == "normal":
        return torch.randn(*shape, device=device, dtype=dtype)
    if distribution == "uniform":
        return 2.0 * torch.rand(*shape, device=device, dtype=dtype) - 1.0
    raise ValueError(f"Unknown distribution: {distribution}")


def dense_random_tensor(
    dims: Sequence[int],
    *,
    distribution: str,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _sample(dims, distribution, device, dtype)


def block_diagonal_tensor(
    dims: Sequence[int],
    *,
    num_blocks: int,
    block_scale: float,
    off_block_scale: float,
    distribution: str,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if int(num_blocks) < 1:
        raise ValueError("num_blocks must be positive")
    if any(int(dim) < int(num_blocks) for dim in dims):
        raise ValueError("Every tensor dimension must be at least num_blocks")

    tensor = float(off_block_scale) * _sample(dims, distribution, device, dtype)
    mode_sizes = []
    for dim in dims:
        base, remainder = divmod(int(dim), int(num_blocks))
        mode_sizes.append([base + (1 if block < remainder else 0) for block in range(int(num_blocks))])

    starts = []
    for sizes in mode_sizes:
        offsets = []
        current = 0
        for size in sizes:
            offsets.append(current)
            current += size
        starts.append(offsets)

    for block in range(int(num_blocks)):
        slices = []
        block_shape = []
        for mode in range(len(dims)):
            start = starts[mode][block]
            size = mode_sizes[mode][block]
            slices.append(slice(start, start + size))
            block_shape.append(size)
        tensor[tuple(slices)] = float(block_scale) * _sample(
            block_shape, distribution, device, dtype
        )
    return tensor


def make_synthetic_tensor(
    data: str,
    dims: Sequence[int],
    *,
    components: int,
    distribution: str,
    num_blocks: int,
    block_scale: float,
    off_block_scale: float,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    data = str(data).lower()
    if data == "random":
        tensor = dense_random_tensor(
            dims,
            distribution=distribution,
            device=device,
            dtype=dtype,
        )
    elif data in {"block", "block_diagonal", "blockdiag"}:
        tensor = block_diagonal_tensor(
            dims,
            num_blocks=num_blocks,
            block_scale=block_scale,
            off_block_scale=off_block_scale,
            distribution=distribution,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError("data must be 'random' or 'block'")

    return (normalize_fro(tensor) * float(components)).contiguous()
