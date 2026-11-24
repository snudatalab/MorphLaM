from __future__ import annotations

"""PuzzleTensor-style hyperslice shifting followed by classical factorization."""

import argparse
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from classical import cp_als, hooi, tt_svd_permutation_search
from common import (
    ProgressCSV,
    TrainResult,
    choose_cp_rank,
    choose_tucker_ranks,
    cp_parameter_count,
    cp_to_tensor,
    count_parameters,
    get_dtype,
    make_result_row,
    normalize_tt_ranks,
    parse_dims,
    parse_int_list,
    prod,
    resolve_budget,
    save_rows,
    set_seed,
    tt_parameter_count,
    tt_to_tensor,
    tucker_parameter_count,
    tucker_to_tensor,
)
from synthetic import make_synthetic_tensor


def split_blocks(tensor: torch.Tensor, block_counts: Sequence[int]) -> torch.Tensor:
    if len(block_counts) != tensor.ndim:
        raise ValueError("block_counts must contain one value per tensor mode")
    if any(int(dim) % int(count) != 0 for dim, count in zip(tensor.shape, block_counts)):
        raise ValueError("Each tensor dimension must be divisible by its block count")
    block_sizes = [int(dim) // int(count) for dim, count in zip(tensor.shape, block_counts)]
    interleaved: List[int] = []
    for count, size in zip(block_counts, block_sizes):
        interleaved.extend([int(count), int(size)])
    reshaped = tensor.reshape(*interleaved)
    order = len(block_counts)
    permutation = [2 * mode for mode in range(order)] + [2 * mode + 1 for mode in range(order)]
    return reshaped.permute(*permutation).contiguous().reshape(prod(block_counts), *block_sizes)


def merge_blocks(blocks: torch.Tensor, original_dims: Sequence[int], block_counts: Sequence[int]) -> torch.Tensor:
    order = len(block_counts)
    block_sizes = [int(dim) // int(count) for dim, count in zip(original_dims, block_counts)]
    expanded = blocks.reshape(*block_counts, *block_sizes)
    permutation: List[int] = []
    for mode in range(order):
        permutation.extend([mode, order + mode])
    return expanded.permute(*permutation).contiguous().reshape(*original_dims)


class PuzzleTransform(nn.Module):
    """Frequency-domain hyperslice shifts following the PuzzleTensor construction."""

    def __init__(
        self,
        shape: Sequence[int],
        num_layers: int,
        random_init: bool,
        *,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.batch = int(shape[0])
        self.shape = tuple(int(value) for value in shape[1:])
        self.order = len(self.shape)
        self.num_layers = int(num_layers)
        if self.order < 2:
            raise ValueError("PuzzleTensor requires at least two tensor modes")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

        initializer = torch.rand if random_init else torch.zeros
        self.shifts = nn.ModuleList()
        for layer in range(self.num_layers):
            current = layer % self.order
            view_shape = (self.batch,) + tuple(-1 if mode == current else 1 for mode in range(self.order))
            parameters = [
                nn.Parameter(
                    initializer(
                        self.batch, self.shape[current], device=device, dtype=dtype
                    ).view(view_shape)
                )
                for _ in range(self.order - 1)
            ]
            self.shifts.append(nn.ParameterList(parameters))

        self.full_phase_names: List[str] = []
        for dimension in range(self.order - 1):
            view_shape = tuple(-1 if mode == dimension else 1 for mode in range(self.order))
            frequencies = (
                torch.arange(self.shape[dimension], device=device, dtype=dtype)
                + self.shape[dimension] // 2
            ) % self.shape[dimension] - self.shape[dimension] // 2
            frequencies[self.shape[dimension] // 2] *= self.shape[dimension] % 2
            name = f"full_phase_{dimension}"
            self.register_buffer(
                name,
                2.0j * torch.pi * frequencies.view(view_shape) / self.shape[dimension],
                persistent=False,
            )
            self.full_phase_names.append(name)

        self.half_phase_names: List[str] = []
        for dimension in range(self.order - 2, self.order):
            view_shape = tuple(-1 if mode == dimension else 1 for mode in range(self.order))
            frequencies = torch.arange(
                self.shape[dimension] // 2 + 1, device=device, dtype=dtype
            )
            frequencies[-1] *= self.shape[dimension] % 2
            name = f"half_phase_{dimension}"
            self.register_buffer(
                name,
                2.0j * torch.pi * frequencies.view(view_shape) / self.shape[dimension],
                persistent=False,
            )
            self.half_phase_names.append(name)

        self.fft_dims: List[Tuple[int, ...]] = []
        self.inverse_sizes: List[Tuple[int, ...]] = []
        for layer in range(self.num_layers):
            current = layer % self.order
            self.fft_dims.append(tuple(mode + 1 for mode in range(self.order) if mode != current))
            self.inverse_sizes.append(tuple(size for mode, size in enumerate(self.shape) if mode != current))

        self.mode_permutations = [
            [0, mode + 1] + [other + 1 for other in range(self.order) if other != mode]
            for mode in range(self.order)
        ]
        self.coshapes = [prod(self.shape) // self.shape[mode] for mode in range(self.order)]
        self.norm_factors = [1.0 / math.sqrt(self.shape[mode]) for mode in range(self.order)]

    def _full_phase(self, index: int) -> torch.Tensor:
        return getattr(self, self.full_phase_names[index])

    def _half_phase(self, index: int) -> torch.Tensor:
        return getattr(self, self.half_phase_names[index])

    def forward(self, value: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        layers = range(self.num_layers) if not reverse else reversed(range(self.num_layers))
        for layer in layers:
            if layer or reverse:
                value = torch.fft.rfftn(value, dim=self.fft_dims[layer])
            current = layer % self.order
            phase = torch.exp(
                self.shifts[layer][self.order - 2]
                * self._half_phase(int(current <= self.order - 2))
            )
            for index in range(self.order - 2):
                phase = phase * torch.exp(
                    self.shifts[layer][index] * self._full_phase(int(current <= index) + index)
                )
            multiplier = torch.conj(phase) if reverse else phase
            value = torch.fft.irfftn(
                multiplier * value,
                s=self.inverse_sizes[layer],
                dim=self.fft_dims[layer],
            )
        return value

    def surrogate_loss(self, value: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), device=value.device, dtype=value.dtype)
        for mode in range(self.order):
            unfolded = value.permute(*self.mode_permutations[mode]).contiguous().reshape(
                self.batch, self.shape[mode], self.coshapes[mode]
            )
            loss = loss + torch.linalg.matrix_norm(unfolded, ord="nuc").sum() * self.norm_factors[mode]
        return loss


def train_transform(
    frequency_blocks: torch.Tensor,
    model: PuzzleTransform,
    *,
    epochs: int,
    learning_rate: float,
    log_every: int,
    seed: int,
    progress: ProgressCSV,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    start = time.time()

    if model.num_layers > 3:
        for layer in range(model.num_layers - 3, model.num_layers):
            for parameter in model.shifts[layer]:
                parameter.requires_grad = False

    for epoch in range(int(epochs)):
        if model.num_layers > 3:
            boundaries = {
                int(epochs) // 4: model.num_layers - 3,
                2 * int(epochs) // 4: model.num_layers - 2,
                3 * int(epochs) // 4: model.num_layers - 1,
            }
            if epoch in boundaries:
                for parameter in model.shifts[boundaries[epoch]]:
                    parameter.requires_grad = True

        transformed = model(frequency_blocks)
        loss = model.surrogate_loss(transformed)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        one_based = epoch + 1
        if one_based == 1 or one_based == int(epochs) or one_based % int(log_every) == 0:
            elapsed = time.time() - start
            loss_value = float(loss.detach().cpu().item())
            print(
                f"[puzzletensor] epoch={one_based:05d}/{epochs} "
                f"surrogate={loss_value:.6e} time={elapsed:.2f}s",
                flush=True,
            )
            progress.append(
                {
                    "seed": seed,
                    "method": "puzzletensor_transform",
                    "epoch": one_based,
                    "total_epochs": epochs,
                    "elapsed_total_sec": elapsed,
                    "current_mse": loss_value,
                    "current_rel_fro": "",
                    "best_rel_fro": "",
                    "best_epoch": "",
                }
            )
    return time.time() - start


def inverse_permutation(permutation: Sequence[int]) -> Tuple[int, ...]:
    inverse = [0] * len(permutation)
    for new_axis, old_axis in enumerate(permutation):
        inverse[int(old_axis)] = int(new_axis)
    return tuple(inverse)


def reconstruct_transformed_blocks(
    blocks: torch.Tensor,
    method: str,
    per_block_budget: int,
    *,
    cp_rank_override: int,
    tucker_rank_override: Optional[List[int]],
    tt_rank_override: Optional[List[int]],
    tt_permutations: str,
    cp_sweeps: int,
    hooi_sweeps: int,
    seed: int,
) -> Tuple[torch.Tensor, int, float, Dict[str, object]]:
    estimates: List[torch.Tensor] = []
    total_solver_time = 0.0
    block_shape = tuple(int(dim) for dim in blocks.shape[1:])

    if method == "cp":
        rank = cp_rank_override if cp_rank_override > 0 else choose_cp_rank(block_shape, per_block_budget)
        per_block_parameters = cp_parameter_count(block_shape, rank)
        for index, block in enumerate(blocks):
            solved = cp_als(block, rank, sweeps=cp_sweeps, seed=seed + index)
            estimates.append(cp_to_tensor(solved.factors))
            total_solver_time += solved.train.train_time_sec
        extras: Dict[str, object] = {"downstream": "cp", "rank": rank}

    elif method == "tucker":
        ranks = tucker_rank_override or choose_tucker_ranks(block_shape, per_block_budget)
        per_block_parameters = tucker_parameter_count(block_shape, ranks)
        for index, block in enumerate(blocks):
            solved = hooi(block, ranks, sweeps=hooi_sweeps, seed=seed + index)
            estimates.append(tucker_to_tensor(solved.core, solved.factors))
            total_solver_time += solved.train.train_time_sec
        extras = {"downstream": "tucker", "ranks": "x".join(map(str, ranks))}

    elif method == "tt":
        per_block_parameters = 0
        selected_ranks: Optional[List[int]] = None
        selected_permutation: Optional[Tuple[int, ...]] = None
        for block in blocks:
            solved = tt_svd_permutation_search(
                block,
                per_block_budget,
                permutations_spec=tt_permutations,
                fixed_ranks=tt_rank_override,
            )
            estimate_permuted = tt_to_tensor(solved.cores)
            estimates.append(estimate_permuted.permute(*inverse_permutation(solved.permutation)).contiguous())
            total_solver_time += solved.train.train_time_sec
            block_parameters = tt_parameter_count(
                [block_shape[index] for index in solved.permutation], solved.ranks
            )
            per_block_parameters = max(per_block_parameters, block_parameters)
            selected_ranks = solved.ranks
            selected_permutation = solved.permutation
        extras = {
            "downstream": "tt",
            "ranks": "x".join(map(str, selected_ranks or [])),
            "permutation": ",".join(map(str, selected_permutation or [])),
        }
    else:
        raise ValueError(f"Unknown downstream decomposition: {method}")

    return torch.stack(estimates), per_block_parameters, total_solver_time, extras


def parse_rank_vector(value: str, expected: int) -> Optional[List[int]]:
    parsed = parse_int_list(value)
    if not parsed:
        return None
    if len(parsed) == 1:
        return parsed * expected
    if len(parsed) != expected:
        raise ValueError(f"Expected one rank or {expected} ranks; got {parsed}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", default="64,64,64")
    parser.add_argument("--data", choices=["random", "block"], default="random")
    parser.add_argument("--distribution", choices=["normal", "uniform"], default="normal")
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--block_scale", type=float, default=1.0)
    parser.add_argument("--off_block_scale", type=float, default=0.0)
    parser.add_argument("--components", type=int, default=24)

    parser.add_argument("--parameter_ratio", type=float, default=0.125)
    parser.add_argument("--target_params", type=int, default=-1)
    parser.add_argument("--block_counts", default="1", help="One count or one count per tensor mode")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--transform_epochs", type=int, default=5000)
    parser.add_argument("--transform_lr", type=float, default=1e-3)
    parser.add_argument("--random_shift_init", action="store_true")
    parser.add_argument("--downstream", choices=["cp", "tucker", "tt", "best"], default="best")

    parser.add_argument("--cp_rank", type=int, default=-1)
    parser.add_argument("--tucker_ranks", default="")
    parser.add_argument("--tt_ranks", default="")
    parser.add_argument("--tt_permutations", default="auto")
    parser.add_argument("--cp_als_sweeps", type=int, default=500)
    parser.add_argument("--hooi_sweeps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--out_dir", default="results/puzzletensor")
    parser.add_argument("--results_csv", default="")
    parser.add_argument("--log_progress_csv", nargs="?", const="auto", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.log_every <= 0:
        raise ValueError("log_every must be positive")
    dims = parse_dims(args.dims)
    block_counts = parse_int_list(args.block_counts)
    if len(block_counts) == 1:
        block_counts = block_counts * len(dims)
    if len(block_counts) != len(dims):
        raise ValueError("block_counts must contain one value or one value per mode")
    dtype = get_dtype(args.dtype)
    budget = resolve_budget(dims, args.parameter_ratio, args.target_params)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = args.results_csv or os.path.join(args.out_dir, "results.csv")
    progress_path = (
        os.path.join(args.out_dir, "progress.csv")
        if args.log_progress_csv == "auto"
        else args.log_progress_csv
    )
    progress = ProgressCSV(progress_path)
    seeds = parse_int_list(args.seeds) or [int(args.seed)]
    rows: List[Dict[str, object]] = []

    for seed in seeds:
        set_seed(seed, args.deterministic)
        target = make_synthetic_tensor(
            args.data,
            dims,
            components=args.components,
            distribution=args.distribution,
            num_blocks=args.num_blocks,
            block_scale=args.block_scale,
            off_block_scale=args.off_block_scale,
            device=args.device,
            dtype=dtype,
        )
        blocks = split_blocks(target, block_counts)
        transform = PuzzleTransform(
            blocks.shape,
            args.num_layers,
            args.random_shift_init,
            device=args.device,
            dtype=dtype,
        )
        transform_parameters = count_parameters(transform)
        remaining_budget = budget - transform_parameters
        if remaining_budget <= 0:
            raise ValueError(
                f"PuzzleTensor shifts use {transform_parameters} parameters, exceeding budget {budget}"
            )
        per_block_budget = remaining_budget // blocks.shape[0]
        if per_block_budget <= 0:
            raise ValueError("No budget remains for the downstream decomposition")

        first_fft_dims = tuple(range(2, blocks.ndim))
        frequency_blocks = torch.fft.rfftn(blocks, dim=first_fft_dims)
        transform_time = train_transform(
            frequency_blocks,
            transform,
            epochs=args.transform_epochs,
            learning_rate=args.transform_lr,
            log_every=args.log_every,
            seed=seed,
            progress=progress,
        )
        with torch.no_grad():
            transformed = transform(frequency_blocks).detach()

        candidate_methods = [args.downstream] if args.downstream != "best" else ["cp", "tucker", "tt"]
        best_row: Optional[Dict[str, object]] = None
        best_rel = float("inf")
        tucker_override = parse_rank_vector(args.tucker_ranks, len(dims))
        tt_override = parse_rank_vector(args.tt_ranks, len(dims) - 1)

        for downstream in candidate_methods:
            reconstructed_transformed, per_block_parameters, solver_time, extras = reconstruct_transformed_blocks(
                transformed,
                downstream,
                per_block_budget,
                cp_rank_override=args.cp_rank,
                tucker_rank_override=tucker_override,
                tt_rank_override=tt_override,
                tt_permutations=args.tt_permutations,
                cp_sweeps=args.cp_als_sweeps,
                hooi_sweeps=args.hooi_sweeps,
                seed=seed,
            )
            with torch.no_grad():
                reconstructed_blocks = transform(reconstructed_transformed, reverse=True)
                estimate = merge_blocks(reconstructed_blocks, dims, block_counts)
                residual = estimate - target
                mse = float(residual.square().mean().detach().cpu().item())
                rel = float(
                    (torch.linalg.norm(residual) / torch.linalg.norm(target).clamp_min(1e-12))
                    .detach()
                    .cpu()
                    .item()
                )
            total_parameters = transform_parameters + int(blocks.shape[0]) * per_block_parameters
            result = TrainResult(
                best_epoch=args.transform_epochs,
                train_time_sec=transform_time + solver_time,
                mse=mse,
                rel_fro=rel,
                final_mse=mse,
            )
            row = make_result_row(
                seed=seed,
                method=f"puzzletensor_{downstream}",
                data=args.data,
                dims=dims,
                budget=budget,
                n_params=total_parameters,
                result=result,
                extras={
                    **extras,
                    "num_layers": args.num_layers,
                    "block_counts": "x".join(map(str, block_counts)),
                    "shift_params": transform_parameters,
                },
            )
            if rel < best_rel:
                best_rel = rel
                best_row = row

        assert best_row is not None
        rows.append(best_row)
        save_rows(rows, results_path)
        print(
            f"[result] seed={seed} method={best_row['method']} "
            f"rel={float(best_row['rel_fro']):.6e} params={best_row['n_params']}/{budget}",
            flush=True,
        )
    print(f"Saved results to {os.path.abspath(results_path)}")


if __name__ == "__main__":
    main()
