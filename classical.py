from __future__ import annotations

"""Classical tensor models and standard dense decomposition solvers."""

import math
import time
from dataclasses import dataclass
from itertools import permutations
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from common import (
    DenseTensorModel,
    TrainResult,
    choose_tt_ranks,
    cp_to_tensor,
    cuda_sync,
    normalize_tt_ranks,
    parse_int_list,
    prod,
    set_seed,
    tt_parameter_count,
    tt_to_tensor,
    tucker_to_tensor,
)


class CPModel(DenseTensorModel):
    def __init__(
        self,
        dims: Sequence[int],
        rank: int,
        *,
        init_scale: float,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.factors = nn.ParameterList(
            [
                nn.Parameter(float(init_scale) * torch.randn(int(dim), self.rank, device=device, dtype=dtype))
                for dim in dims
            ]
        )

    def reconstruct(self) -> torch.Tensor:
        return cp_to_tensor(list(self.factors))

    def cp_factors(self) -> List[torch.Tensor]:
        return list(self.factors)


class TuckerModel(DenseTensorModel):
    def __init__(
        self,
        dims: Sequence[int],
        ranks: Sequence[int],
        *,
        init_scale: float,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.ranks = tuple(int(rank) for rank in ranks)
        if len(self.ranks) != len(dims):
            raise ValueError("Tucker requires one rank per mode")
        self.core = nn.Parameter(float(init_scale) * torch.randn(*self.ranks, device=device, dtype=dtype))
        self.factors = nn.ParameterList(
            [
                nn.Parameter(float(init_scale) * torch.randn(int(dim), rank, device=device, dtype=dtype))
                for dim, rank in zip(dims, self.ranks)
            ]
        )

    def reconstruct(self) -> torch.Tensor:
        return tucker_to_tensor(self.core, list(self.factors))


class TTModel(DenseTensorModel):
    def __init__(
        self,
        dims: Sequence[int],
        ranks: Sequence[int],
        *,
        init_scale: float,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        dims = tuple(int(dim) for dim in dims)
        self.ranks = tuple(normalize_tt_ranks(dims, ranks))
        full = [1, *self.ranks, 1]
        cores: List[nn.Parameter] = []
        for mode, dim in enumerate(dims):
            shape = (full[mode], dim, full[mode + 1])
            core = float(init_scale) * torch.randn(*shape, device=device, dtype=dtype)
            if mode == 0:
                core = core.reshape(dim, full[1])
            elif mode == len(dims) - 1:
                core = core.reshape(full[-2], dim)
            cores.append(nn.Parameter(core))
        self.cores = nn.ParameterList(cores)

    def reconstruct(self) -> torch.Tensor:
        return tt_to_tensor(list(self.cores))


@dataclass
class CPALSResult:
    factors: List[torch.Tensor]
    train: TrainResult
    restart: int


@dataclass
class HOOIResult:
    core: torch.Tensor
    factors: List[torch.Tensor]
    train: TrainResult
    restart: int


@dataclass
class TTSVDResult:
    cores: List[torch.Tensor]
    ranks: List[int]
    permutation: Tuple[int, ...]
    train: TrainResult


def _labels(required: int) -> List[str]:
    import string

    labels = list(string.ascii_lowercase + string.ascii_uppercase)
    if required > len(labels):
        raise ValueError("Tensor order is too large for the einsum helper")
    return labels[:required]


def mode_unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    return tensor.movedim(int(mode), 0).contiguous().reshape(tensor.shape[int(mode)], -1)


def leading_left_singular_vectors(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    rank = max(1, min(int(rank), min(matrix.shape)))
    work = matrix if matrix.dtype in (torch.float32, torch.float64) else matrix.float()
    vectors, _, _ = torch.linalg.svd(work, full_matrices=False)
    return vectors[:, :rank].to(device=matrix.device, dtype=matrix.dtype)


def _mttkrp(tensor: torch.Tensor, factors: Sequence[torch.Tensor], mode: int) -> torch.Tensor:
    labels = _labels(tensor.ndim + 1)
    tensor_labels = labels[: tensor.ndim]
    rank_label = labels[tensor.ndim]
    lhs = ["".join(tensor_labels)]
    operands: List[torch.Tensor] = [tensor]
    for other_mode in range(tensor.ndim):
        if other_mode == int(mode):
            continue
        lhs.append(tensor_labels[other_mode] + rank_label)
        operands.append(factors[other_mode])
    rhs = tensor_labels[int(mode)] + rank_label
    return torch.einsum(f"{','.join(lhs)}->{rhs}", *operands)


def _balance_cp_columns(factors: List[torch.Tensor], eps: float = 1e-12) -> None:
    with torch.no_grad():
        weights = torch.ones(factors[0].shape[1], device=factors[0].device, dtype=factors[0].dtype)
        for factor in factors:
            norms = torch.linalg.norm(factor, dim=0).clamp_min(float(eps))
            factor.div_(norms)
            weights.mul_(norms)
        shared = weights.clamp_min(float(eps)).pow(1.0 / len(factors))
        for factor in factors:
            factor.mul_(shared)


def _cp_initialization(
    tensor: torch.Tensor,
    rank: int,
    initialization: str,
    init_scale: float,
    seed: int,
) -> List[torch.Tensor]:
    set_seed(seed)
    factors: List[torch.Tensor] = []
    for mode, dim in enumerate(tensor.shape):
        if initialization == "svd":
            available = min(int(dim), int(rank))
            factor = leading_left_singular_vectors(mode_unfold(tensor, mode), available)
            if available < int(rank):
                extra = float(init_scale) * torch.randn(
                    int(dim), int(rank) - available, device=tensor.device, dtype=tensor.dtype
                )
                factor = torch.cat([factor, extra], dim=1)
            factor = factor + 0.01 * float(init_scale) * torch.randn_like(factor)
        elif initialization == "random":
            factor = float(init_scale) * torch.randn(
                int(dim), int(rank), device=tensor.device, dtype=tensor.dtype
            )
        else:
            raise ValueError(f"Unknown CP-ALS initialization: {initialization}")
        factors.append(factor.contiguous())
    _balance_cp_columns(factors)
    return factors


def cp_als(
    tensor: torch.Tensor,
    rank: int,
    *,
    sweeps: int = 500,
    tolerance: float = 1e-7,
    patience: int = 5,
    regularization: float = 1e-7,
    restarts: int = 3,
    init_scale: float = 0.02,
    seed: int = 0,
    verbose: bool = False,
) -> CPALSResult:
    rank = int(rank)
    if rank <= 0:
        raise ValueError("CP rank must be positive")
    cuda_sync(tensor.device)
    start = time.time()
    target_norm = float(torch.linalg.norm(tensor).detach().double().cpu().item())
    best_rel = float("inf")
    best_mse = float("inf")
    best_sweep = -1
    best_restart = -1
    best_factors: Optional[List[torch.Tensor]] = None

    for restart in range(max(1, int(restarts))):
        factors = _cp_initialization(
            tensor,
            rank,
            "svd" if restart == 0 else "random",
            init_scale,
            int(seed) + 1009 * restart,
        )
        previous = float("inf")
        stable = 0
        for sweep in range(1, int(sweeps) + 1):
            for mode in range(tensor.ndim):
                gram = torch.ones(rank, rank, device=tensor.device, dtype=tensor.dtype)
                for other_mode, factor in enumerate(factors):
                    if other_mode != mode:
                        gram = gram * (factor.transpose(0, 1) @ factor)
                gram = gram + float(regularization) * torch.eye(
                    rank, device=tensor.device, dtype=tensor.dtype
                )
                rhs = _mttkrp(tensor, factors, mode)
                try:
                    factors[mode] = torch.linalg.solve(
                        gram.transpose(0, 1), rhs.transpose(0, 1)
                    ).transpose(0, 1).contiguous()
                except RuntimeError:
                    factors[mode] = (rhs @ torch.linalg.pinv(gram)).contiguous()
            _balance_cp_columns(factors)

            with torch.no_grad():
                residual = cp_to_tensor(factors) - tensor
                mse = float(residual.square().mean().detach().double().cpu().item())
                rel = float(torch.linalg.norm(residual).detach().double().cpu().item()) / max(
                    target_norm, 1e-30
                )
            if verbose and (sweep == 1 or sweep % 10 == 0):
                print(f"[cp_als] restart={restart} sweep={sweep} rel={rel:.6e}")
            if rel < best_rel:
                best_rel = rel
                best_mse = mse
                best_sweep = sweep
                best_restart = restart
                best_factors = [factor.detach().clone() for factor in factors]

            relative_change = abs(previous - rel) / max(abs(previous), 1.0) if math.isfinite(previous) else float("inf")
            stable = stable + 1 if relative_change <= float(tolerance) else 0
            previous = rel
            if stable >= max(1, int(patience)):
                break

    if best_factors is None:
        raise RuntimeError("CP-ALS failed to produce a finite solution")
    cuda_sync(tensor.device)
    train = TrainResult(
        best_epoch=best_sweep,
        train_time_sec=time.time() - start,
        mse=best_mse,
        rel_fro=best_rel,
        final_mse=best_mse,
    )
    return CPALSResult(best_factors, train, best_restart)


def _tucker_core(tensor: torch.Tensor, factors: Sequence[torch.Tensor]) -> torch.Tensor:
    order = tensor.ndim
    labels = _labels(2 * order)
    observed = labels[:order]
    latent = labels[order:]
    lhs = ["".join(observed)] + [observed[mode] + latent[mode] for mode in range(order)]
    return torch.einsum(f"{','.join(lhs)}->{''.join(latent)}", tensor, *factors)


def _project_except(
    tensor: torch.Tensor, factors: Sequence[torch.Tensor], mode: int
) -> torch.Tensor:
    order = tensor.ndim
    labels = _labels(2 * order)
    observed = labels[:order]
    latent = labels[order:]
    lhs = ["".join(observed)]
    operands: List[torch.Tensor] = [tensor]
    for other_mode in range(order):
        if other_mode == int(mode):
            continue
        lhs.append(observed[other_mode] + latent[other_mode])
        operands.append(factors[other_mode])
    rhs = observed[int(mode)] + "".join(
        latent[other_mode] for other_mode in range(order) if other_mode != int(mode)
    )
    return torch.einsum(f"{','.join(lhs)}->{rhs}", *operands)


def _hosvd_factors(tensor: torch.Tensor, ranks: Sequence[int]) -> List[torch.Tensor]:
    return [
        leading_left_singular_vectors(mode_unfold(tensor, mode), int(ranks[mode]))
        for mode in range(tensor.ndim)
    ]


def _random_orthogonal_factors(
    tensor: torch.Tensor, ranks: Sequence[int], seed: int
) -> List[torch.Tensor]:
    set_seed(seed)
    factors: List[torch.Tensor] = []
    for dim, rank in zip(tensor.shape, ranks):
        matrix = torch.randn(int(dim), int(rank), device=tensor.device, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        factors.append(q[:, : int(rank)].contiguous())
    return factors


def hooi(
    tensor: torch.Tensor,
    ranks: Sequence[int],
    *,
    sweeps: int = 100,
    tolerance: float = 1e-7,
    patience: int = 3,
    restarts: int = 2,
    seed: int = 0,
    verbose: bool = False,
) -> HOOIResult:
    ranks = [int(rank) for rank in ranks]
    if len(ranks) != tensor.ndim:
        raise ValueError("HOOI requires one multilinear rank per mode")
    if any(rank <= 0 or rank > int(dim) for rank, dim in zip(ranks, tensor.shape)):
        raise ValueError(f"Invalid Tucker ranks {ranks} for shape {tuple(tensor.shape)}")

    cuda_sync(tensor.device)
    start = time.time()
    target_sq = float(tensor.square().sum().detach().double().cpu().item())
    target_norm = math.sqrt(target_sq)
    best_rel = float("inf")
    best_mse = float("inf")
    best_sweep = -1
    best_restart = -1
    best_core: Optional[torch.Tensor] = None
    best_factors: Optional[List[torch.Tensor]] = None

    for restart in range(max(1, int(restarts))):
        factors = (
            _hosvd_factors(tensor, ranks)
            if restart == 0
            else _random_orthogonal_factors(tensor, ranks, int(seed) + 1297 * restart)
        )
        previous = float("inf")
        stable = 0
        for sweep in range(1, int(sweeps) + 1):
            for mode in range(tensor.ndim):
                projected = _project_except(tensor, factors, mode)
                factors[mode] = leading_left_singular_vectors(
                    projected.reshape(int(tensor.shape[mode]), -1), ranks[mode]
                ).contiguous()
            with torch.no_grad():
                core = _tucker_core(tensor, factors)
                residual_sq = max(
                    0.0,
                    target_sq - float(core.square().sum().detach().double().cpu().item()),
                )
                rel = math.sqrt(residual_sq) / max(target_norm, 1e-30)
                mse = residual_sq / tensor.numel()
            if verbose and (sweep == 1 or sweep % 10 == 0):
                print(f"[hooi] restart={restart} sweep={sweep} rel={rel:.6e}")
            if rel < best_rel:
                best_rel = rel
                best_mse = mse
                best_sweep = sweep
                best_restart = restart
                best_core = core.detach().clone()
                best_factors = [factor.detach().clone() for factor in factors]

            relative_change = abs(previous - rel) / max(abs(previous), 1.0) if math.isfinite(previous) else float("inf")
            stable = stable + 1 if relative_change <= float(tolerance) else 0
            previous = rel
            if stable >= max(1, int(patience)):
                break

    if best_core is None or best_factors is None:
        raise RuntimeError("HOOI failed to produce a finite solution")
    cuda_sync(tensor.device)
    train = TrainResult(
        best_epoch=best_sweep,
        train_time_sec=time.time() - start,
        mse=best_mse,
        rel_fro=best_rel,
        final_mse=best_mse,
    )
    return HOOIResult(best_core, best_factors, train, best_restart)


def tt_svd_decompose(tensor: torch.Tensor, ranks: Sequence[int]) -> List[torch.Tensor]:
    dims = tuple(int(dim) for dim in tensor.shape)
    ranks = normalize_tt_ranks(dims, ranks)
    cores: List[torch.Tensor] = []
    residual = tensor
    previous_rank = 1
    for mode in range(len(dims) - 1):
        matrix = residual.reshape(previous_rank * dims[mode], -1)
        work = matrix if matrix.dtype in (torch.float32, torch.float64) else matrix.float()
        u, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
        next_rank = min(int(ranks[mode]), int(u.shape[1]))
        u = u[:, :next_rank].to(device=tensor.device, dtype=tensor.dtype)
        singular_values = singular_values[:next_rank].to(device=tensor.device, dtype=tensor.dtype)
        vh = vh[:next_rank, :].to(device=tensor.device, dtype=tensor.dtype)
        if mode == 0:
            cores.append(u.reshape(dims[mode], next_rank).contiguous())
        else:
            cores.append(u.reshape(previous_rank, dims[mode], next_rank).contiguous())
        residual = (singular_values[:, None] * vh).contiguous()
        previous_rank = next_rank
    cores.append(residual.reshape(previous_rank, dims[-1]).contiguous())
    return cores


def parse_tt_permutations(specification: str, order: int) -> List[Tuple[int, ...]]:
    specification = str(specification or "auto").strip().lower()
    natural = tuple(range(int(order)))
    if specification in {"natural", "identity", "none"}:
        return [natural]
    if specification in {"natural_reverse", "two"}:
        return list(dict.fromkeys([natural, tuple(reversed(natural))]))
    if specification == "all":
        return [tuple(candidate) for candidate in permutations(range(int(order)))]
    if specification == "auto":
        if int(order) <= 4:
            return [tuple(candidate) for candidate in permutations(range(int(order)))]
        candidates = [natural, tuple(reversed(natural))]
        for shift in range(1, int(order)):
            candidates.append(tuple(list(natural[shift:]) + list(natural[:shift])))
        return list(dict.fromkeys(candidates))

    output: List[Tuple[int, ...]] = []
    for part in specification.split("|"):
        candidate = tuple(parse_int_list(part))
        if sorted(candidate) != list(range(int(order))):
            raise ValueError(f"Invalid TT permutation {candidate}")
        output.append(candidate)
    return list(dict.fromkeys(output))


def tt_svd_permutation_search(
    tensor: torch.Tensor,
    budget: int,
    *,
    permutations_spec: str = "auto",
    fixed_ranks: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> TTSVDResult:
    cuda_sync(tensor.device)
    start = time.time()
    best: Optional[TTSVDResult] = None
    for permutation in parse_tt_permutations(permutations_spec, tensor.ndim):
        permuted = tensor.permute(*permutation).contiguous()
        dims = tuple(int(dim) for dim in permuted.shape)
        ranks = (
            normalize_tt_ranks(dims, fixed_ranks)
            if fixed_ranks is not None
            else choose_tt_ranks(dims, int(budget))
        )
        if tt_parameter_count(dims, ranks) > int(budget):
            continue
        local_start = time.time()
        cores = tt_svd_decompose(permuted, ranks)
        with torch.no_grad():
            residual = tt_to_tensor(cores) - permuted
            mse = float(residual.square().mean().detach().double().cpu().item())
            rel = float(
                torch.linalg.norm(residual).detach().double().cpu().item()
                / max(float(torch.linalg.norm(permuted).detach().double().cpu().item()), 1e-30)
            )
        if verbose:
            print(f"[tt_svd_perm] permutation={permutation} ranks={ranks} rel={rel:.6e}")
        result = TTSVDResult(
            cores=[core.detach().clone() for core in cores],
            ranks=[int(rank) for rank in ranks],
            permutation=tuple(int(index) for index in permutation),
            train=TrainResult(
                best_epoch=0,
                train_time_sec=time.time() - local_start,
                mse=mse,
                rel_fro=rel,
                final_mse=mse,
            ),
        )
        if best is None or result.train.rel_fro < best.train.rel_fro:
            best = result
    if best is None:
        raise RuntimeError("No feasible TT-SVD configuration was found")
    best.train.train_time_sec = time.time() - start
    cuda_sync(tensor.device)
    return best
