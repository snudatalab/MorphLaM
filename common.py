from __future__ import annotations

"""Shared utilities for the synthetic MorphLaM benchmarks."""

import csv
import math
import os
import string
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def cuda_sync(device: str | torch.device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def parse_int_list(value: str) -> List[int]:
    value = str(value or "").strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> List[float]:
    value = str(value or "").strip()
    if not value:
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> List[str]:
    value = str(value or "").strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_dims(value: str) -> Tuple[int, ...]:
    dims = tuple(parse_int_list(value))
    if len(dims) < 2:
        raise ValueError("--dims must contain at least two positive integers")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"All dimensions must be positive; got {dims}")
    return dims


def get_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[normalized]


def prod(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def normalize_fro(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor).clamp_min(float(eps))


@torch.no_grad()
def relative_fro_error(estimate: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.linalg.norm(estimate - target)
    denominator = torch.linalg.norm(target).clamp_min(1e-12)
    return float((numerator / denominator).detach().float().item())


# -----------------------------------------------------------------------------
# Dense tensor primitives
# -----------------------------------------------------------------------------


def _einsum_labels(required: int) -> List[str]:
    labels = list(string.ascii_lowercase + string.ascii_uppercase)
    if required > len(labels):
        raise ValueError(f"Too many tensor modes for einsum: need {required}")
    return labels[:required]


def cp_to_tensor(factors: Sequence[torch.Tensor]) -> torch.Tensor:
    order = len(factors)
    if order < 2:
        raise ValueError("CP reconstruction requires at least two modes")
    if order == 2:
        return factors[0] @ factors[1].transpose(0, 1)
    if order == 3:
        return torch.einsum("ir,jr,kr->ijk", factors[0], factors[1], factors[2])
    labels = _einsum_labels(order + 1)
    mode_labels = labels[:order]
    rank_label = labels[order]
    lhs = ",".join(f"{mode_labels[index]}{rank_label}" for index in range(order))
    rhs = "".join(mode_labels)
    return torch.einsum(f"{lhs}->{rhs}", *factors)


def tucker_to_tensor(core: torch.Tensor, factors: Sequence[torch.Tensor]) -> torch.Tensor:
    order = len(factors)
    if order == 2:
        return factors[0] @ core @ factors[1].transpose(0, 1)
    labels = _einsum_labels(2 * order)
    output_labels = labels[:order]
    latent_labels = labels[order:]
    core_subscript = "".join(latent_labels)
    factor_subscripts = [f"{output_labels[index]}{latent_labels[index]}" for index in range(order)]
    return torch.einsum(
        f"{core_subscript},{','.join(factor_subscripts)}->{''.join(output_labels)}",
        core,
        *factors,
    )


def tt_to_tensor(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    order = len(cores)
    if order < 2:
        raise ValueError("TT reconstruction requires at least two modes")
    if order == 2:
        return cores[0] @ cores[1]
    result = cores[0]
    for core in cores[1:-1]:
        result = torch.tensordot(result, core, dims=([-1], [0]))
    return torch.tensordot(result, cores[-1], dims=([-1], [0]))


def khatri_rao(factors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not factors:
        raise ValueError("At least one factor is required")
    result = factors[0]
    for factor in factors[1:]:
        result = (
            result[:, None, :] * factor[None, :, :]
        ).reshape(result.shape[0] * factor.shape[0], result.shape[1])
    return result.contiguous()


def choose_balanced_split(dims: Sequence[int]) -> List[int]:
    """Return one side of a product-balanced nontrivial mode bipartition."""
    order = len(dims)
    if order < 2:
        raise ValueError("A bipartition requires at least two modes")
    total = prod(dims)
    best: Optional[List[int]] = None
    best_score = float("inf")
    for size in range(1, order):
        for suffix in combinations(range(1, order), size - 1):
            left = [0, *suffix]
            left_product = prod([dims[index] for index in left])
            right_product = total // left_product
            score = abs(math.log(max(left_product, 1)) - math.log(max(right_product, 1)))
            if score < best_score:
                best = left
                best_score = score
    if best is None:
        raise RuntimeError("Could not construct a balanced split")
    return best


def unfold_tensor(tensor: torch.Tensor, left_modes: Sequence[int]) -> torch.Tensor:
    left = [int(mode) for mode in left_modes]
    right = [mode for mode in range(tensor.ndim) if mode not in left]
    permutation = left + right
    left_size = prod([tensor.shape[mode] for mode in left])
    right_size = prod([tensor.shape[mode] for mode in right])
    return tensor.permute(*permutation).contiguous().reshape(left_size, right_size)


def cp_to_unfolding(factors: Sequence[torch.Tensor], left_modes: Sequence[int]) -> torch.Tensor:
    left = [int(mode) for mode in left_modes]
    right = [mode for mode in range(len(factors)) if mode not in left]
    return khatri_rao([factors[mode] for mode in left]) @ khatri_rao(
        [factors[mode] for mode in right]
    ).transpose(0, 1)


# -----------------------------------------------------------------------------
# Parameter counts and budget allocation
# -----------------------------------------------------------------------------


def resolve_budget(dims: Sequence[int], parameter_ratio: float, target_params: int) -> int:
    if int(target_params) > 0:
        return int(target_params)
    if not (0.0 < float(parameter_ratio) <= 1.0):
        raise ValueError("--parameter_ratio must lie in (0, 1]")
    return max(1, int(math.floor(float(parameter_ratio) * prod(dims))))


def morphlam_parameter_count(
    dims: Sequence[int],
    rank: int,
    components: int,
    num_basis: int,
    shared_profile: bool,
    shared_omega: bool,
) -> int:
    order = len(dims)
    total_mode_size = sum(int(dim) for dim in dims)
    factors = int(rank) * total_mode_size
    shifts = int(components) * total_mode_size
    omega = int(components) if shared_omega else order * int(components)
    profile = int(num_basis) + 2 if shared_profile else order * (int(num_basis) + 2)
    return int(factors + shifts + omega + profile)


def choose_morphlam_components(
    dims: Sequence[int],
    rank: int,
    budget: int,
    num_basis: int,
    shared_profile: bool,
    shared_omega: bool,
) -> int:
    order = len(dims)
    total_mode_size = sum(int(dim) for dim in dims)
    profile = int(num_basis) + 2 if shared_profile else order * (int(num_basis) + 2)
    per_component = total_mode_size + (1 if shared_omega else order)
    remaining = int(budget) - int(rank) * total_mode_size - profile
    if remaining < per_component:
        raise ValueError(
            f"Budget {budget} is too small for rank={rank} and one MorphLaM component"
        )
    return max(1, remaining // per_component)


def cp_parameter_count(dims: Sequence[int], rank: int) -> int:
    return int(sum(int(dim) for dim in dims) * int(rank))


def tucker_parameter_count(dims: Sequence[int], ranks: Sequence[int]) -> int:
    return int(
        sum(int(dim) * int(rank) for dim, rank in zip(dims, ranks))
        + prod([int(rank) for rank in ranks])
    )


def normalize_tt_ranks(dims: Sequence[int], ranks: Sequence[int] | int) -> List[int]:
    if isinstance(ranks, int):
        normalized = [int(ranks)] * (len(dims) - 1)
    else:
        normalized = [int(rank) for rank in ranks]
    if len(normalized) != len(dims) - 1:
        raise ValueError(f"Expected {len(dims) - 1} TT ranks; got {normalized}")
    if any(rank <= 0 for rank in normalized):
        raise ValueError(f"TT ranks must be positive; got {normalized}")
    return normalized


def tt_parameter_count(dims: Sequence[int], ranks: Sequence[int] | int) -> int:
    internal = normalize_tt_ranks(dims, ranks)
    full = [1, *internal, 1]
    return int(
        sum(int(dims[mode]) * full[mode] * full[mode + 1] for mode in range(len(dims)))
    )


def choose_cp_rank(dims: Sequence[int], budget: int) -> int:
    rank = int(budget) // max(1, sum(int(dim) for dim in dims))
    if rank < 1:
        raise ValueError("The budget is too small for rank-one CP")
    return int(rank)


def _largest_uniform_rank(upper: int, fits: Callable[[int], bool]) -> int:
    if upper < 1 or not fits(1):
        raise ValueError("The budget is too small for the minimum rank configuration")
    low, high = 1, int(upper)
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return int(low)


def _maximize_rank_vector(
    initial: Sequence[int],
    caps: Sequence[int],
    budget: int,
    count_fn: Callable[[Sequence[int]], int],
) -> List[int]:
    initial_state = tuple(int(value) for value in initial)
    cap_state = tuple(int(value) for value in caps)
    if count_fn(initial_state) > int(budget):
        raise ValueError("Initial rank vector exceeds the budget")
    best = initial_state
    best_count = count_fn(initial_state)
    stack = [initial_state]
    visited = {initial_state}
    while stack:
        state = stack.pop()
        for index in range(len(state)):
            if state[index] >= cap_state[index]:
                continue
            candidate = list(state)
            candidate[index] += 1
            candidate_state = tuple(candidate)
            if candidate_state in visited:
                continue
            candidate_count = count_fn(candidate_state)
            if candidate_count > int(budget):
                continue
            visited.add(candidate_state)
            stack.append(candidate_state)
            if candidate_count > best_count or (
                candidate_count == best_count and candidate_state < best
            ):
                best = candidate_state
                best_count = candidate_count
    return list(best)


def choose_tucker_ranks(dims: Sequence[int], budget: int) -> List[int]:
    caps = [int(dim) for dim in dims]
    uniform = _largest_uniform_rank(
        min(caps),
        lambda rank: tucker_parameter_count(dims, [rank] * len(dims)) <= int(budget),
    )
    return _maximize_rank_vector(
        [uniform] * len(dims),
        caps,
        int(budget),
        lambda ranks: tucker_parameter_count(dims, ranks),
    )


def tt_rank_caps(dims: Sequence[int]) -> List[int]:
    total = prod(dims)
    left = 1
    caps: List[int] = []
    for bond in range(len(dims) - 1):
        left *= int(dims[bond])
        caps.append(max(1, min(left, total // left)))
    return caps


def choose_tt_ranks(dims: Sequence[int], budget: int) -> List[int]:
    caps = tt_rank_caps(dims)
    uniform = _largest_uniform_rank(
        min(caps),
        lambda rank: tt_parameter_count(dims, [rank] * (len(dims) - 1)) <= int(budget),
    )
    return _maximize_rank_vector(
        [uniform] * (len(dims) - 1),
        caps,
        int(budget),
        lambda ranks: tt_parameter_count(dims, ranks),
    )


# -----------------------------------------------------------------------------
# Training and logging
# -----------------------------------------------------------------------------


class DenseTensorModel(nn.Module):
    def reconstruct(self) -> torch.Tensor:
        raise NotImplementedError

    def apply_post_step_constraints_(self) -> None:
        return None


@dataclass
class TrainResult:
    best_epoch: int
    train_time_sec: float
    mse: float
    rel_fro: float
    final_mse: float


class ProgressCSV:
    fieldnames = [
        "seed",
        "method",
        "epoch",
        "total_epochs",
        "elapsed_total_sec",
        "current_mse",
        "current_rel_fro",
        "best_rel_fro",
        "best_epoch",
    ]

    def __init__(self, path: str):
        self.path = str(path or "").strip()
        if self.path:
            ensure_dir(os.path.dirname(os.path.abspath(self.path)))
            with open(self.path, "w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    def append(self, row: Dict[str, object]) -> None:
        if not self.path:
            return
        payload = {key: row.get(key, "") for key in self.fieldnames}
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(payload)


def _auto_loss(model: DenseTensorModel, target: torch.Tensor) -> torch.Tensor:
    cp_factors_fn = getattr(model, "cp_factors", None)
    if callable(cp_factors_fn) and target.ndim >= 4:
        factors = cp_factors_fn()
        left_modes = choose_balanced_split(tuple(target.shape))
        residual = cp_to_unfolding(factors, left_modes) - unfold_tensor(target, left_modes)
        return residual.square().mean()
    residual = model.reconstruct() - target
    return residual.square().mean()


@torch.no_grad()
def evaluate_model(model: DenseTensorModel, target: torch.Tensor) -> Tuple[float, float]:
    cp_factors_fn = getattr(model, "cp_factors", None)
    if callable(cp_factors_fn) and target.ndim >= 4:
        factors = cp_factors_fn()
        left_modes = choose_balanced_split(tuple(target.shape))
        estimate = cp_to_unfolding(factors, left_modes)
        reference = unfold_tensor(target, left_modes)
    else:
        estimate = model.reconstruct()
        reference = target
    residual = estimate - reference
    mse = float(residual.square().mean().detach().float().item())
    rel = float(
        (torch.linalg.norm(residual) / torch.linalg.norm(reference).clamp_min(1e-12))
        .detach()
        .float()
        .item()
    )
    return mse, rel


def learning_rate_multiplier(epoch: int, total_epochs: int, schedule: str, floor: float) -> float:
    schedule = str(schedule).lower()
    if schedule == "none":
        return 1.0
    if schedule == "cosine_floor":
        progress = min(max(epoch / float(max(total_epochs - 1, 1)), 0.0), 1.0)
        return float(floor) + (1.0 - float(floor)) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown learning-rate schedule: {schedule}")


def train_model(
    model: DenseTensorModel,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    method: str,
    seed: int,
    log_every: int,
    save_every: int,
    lr_schedule: str,
    lr_floor: float,
    progress: Optional[ProgressCSV] = None,
    pre_epoch: Optional[Callable[[int, int], None]] = None,
) -> TrainResult:
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", float(group["lr"]))

    best_rel = float("inf")
    best_mse = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    start = time.time()
    device = str(target.device)

    for epoch in range(int(epochs)):
        if pre_epoch is not None:
            pre_epoch(epoch, int(epochs))
        multiplier = learning_rate_multiplier(epoch, int(epochs), lr_schedule, lr_floor)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * multiplier

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _auto_loss(model, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in {method} at epoch {epoch + 1}")
        loss.backward()
        optimizer.step()
        model.apply_post_step_constraints_()

        one_based = epoch + 1
        should_evaluate = (
            one_based == 1
            or one_based == int(epochs)
            or (int(save_every) > 0 and one_based % int(save_every) == 0)
            or (int(log_every) > 0 and one_based % int(log_every) == 0)
        )
        if should_evaluate:
            mse, rel = evaluate_model(model, target)
            if rel < best_rel:
                best_rel = rel
                best_mse = mse
                best_epoch = one_based
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            if int(log_every) > 0 and (
                one_based == 1 or one_based == int(epochs) or one_based % int(log_every) == 0
            ):
                elapsed = time.time() - start
                print(
                    f"[{method}] epoch={one_based:05d}/{epochs} "
                    f"rel={rel:.6e} best={best_rel:.6e}@{best_epoch} "
                    f"mse={mse:.6e} time={elapsed:.2f}s",
                    flush=True,
                )
                if progress is not None:
                    progress.append(
                        {
                            "seed": int(seed),
                            "method": method,
                            "epoch": one_based,
                            "total_epochs": int(epochs),
                            "elapsed_total_sec": elapsed,
                            "current_mse": mse,
                            "current_rel_fro": rel,
                            "best_rel_fro": best_rel,
                            "best_epoch": best_epoch,
                        }
                    )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_mse, final_rel = evaluate_model(model, target)
    if best_epoch < 0:
        best_epoch = int(epochs)
        best_rel = final_rel
        best_mse = final_mse
    cuda_sync(device)
    return TrainResult(
        best_epoch=int(best_epoch),
        train_time_sec=float(time.time() - start),
        mse=float(best_mse),
        rel_fro=float(best_rel),
        final_mse=float(final_mse),
    )


def save_rows(rows: Sequence[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    columns: List[str] = []
    seen = set()
    preferred = [
        "seed",
        "method",
        "data",
        "dims",
        "budget",
        "n_params",
        "rel_fro",
        "mse",
        "best_epoch",
        "train_time_sec",
    ]
    for column in preferred:
        if any(column in row for row in rows):
            columns.append(column)
            seen.add(column)
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def make_result_row(
    *,
    seed: int,
    method: str,
    data: str,
    dims: Sequence[int],
    budget: int,
    n_params: int,
    result: TrainResult,
    extras: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "seed": int(seed),
        "method": str(method),
        "data": str(data),
        "dims": "x".join(str(int(dim)) for dim in dims),
        "budget": int(budget),
        "n_params": int(n_params),
        "rel_fro": float(result.rel_fro),
        "mse": float(result.mse),
        "best_epoch": int(result.best_epoch),
        "train_time_sec": float(result.train_time_sec),
    }
    if extras:
        row.update(extras)
    return row
