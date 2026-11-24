from __future__ import annotations

"""Hadamard-product CP baseline on synthetic tensors."""

import argparse
import os
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from common import (
    DenseTensorModel,
    ProgressCSV,
    count_parameters,
    cp_to_tensor,
    get_dtype,
    make_result_row,
    parse_dims,
    parse_int_list,
    resolve_budget,
    save_rows,
    set_seed,
    train_model,
)
from synthetic import make_synthetic_tensor


class MultiplicativeCP(DenseTensorModel):
    def __init__(
        self,
        dims: Sequence[int],
        rank_left: int,
        rank_right: int,
        *,
        init_scale: float,
        identity_right_init: bool,
        right_noise_std: float,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank_left = int(rank_left)
        self.rank_right = int(rank_right)
        if self.rank_left < 1 or self.rank_right < 1:
            raise ValueError("Both CP ranks must be positive")
        self.left_factors = nn.ParameterList(
            [
                nn.Parameter(float(init_scale) * torch.randn(int(dim), self.rank_left, device=device, dtype=dtype))
                for dim in dims
            ]
        )
        if identity_right_init:
            base = self.rank_right ** (-1.0 / len(dims))
            self.right_factors = nn.ParameterList(
                [
                    nn.Parameter(
                        base * torch.ones(int(dim), self.rank_right, device=device, dtype=dtype)
                        + float(right_noise_std)
                        * torch.randn(int(dim), self.rank_right, device=device, dtype=dtype)
                    )
                    for dim in dims
                ]
            )
        else:
            self.right_factors = nn.ParameterList(
                [
                    nn.Parameter(float(init_scale) * torch.randn(int(dim), self.rank_right, device=device, dtype=dtype))
                    for dim in dims
                ]
            )

    def reconstruct(self) -> torch.Tensor:
        return cp_to_tensor(list(self.left_factors)) * cp_to_tensor(list(self.right_factors))


def choose_rank_pair(dims: Sequence[int], budget: int, left: int, right: int) -> tuple[int, int]:
    per_rank = sum(int(dim) for dim in dims)
    if left > 0 and right > 0:
        if (left + right) * per_rank > int(budget):
            raise ValueError("The requested rank pair exceeds the parameter budget")
        return int(left), int(right)
    total = int(budget) // max(1, per_rank)
    if total < 2:
        raise ValueError("The budget is too small for rank-(1,1) multiplicative CP")
    rank_left = total // 2
    rank_right = total - rank_left
    return max(1, rank_left), max(1, rank_right)


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
    parser.add_argument("--rank_left", type=int, default=-1)
    parser.add_argument("--rank_right", type=int, default=-1)
    parser.add_argument("--identity_right_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--right_noise_std", type=float, default=1e-3)
    parser.add_argument("--init_scale", type=float, default=0.02)

    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_schedule", choices=["none", "cosine_floor"], default="none")
    parser.add_argument("--lr_floor", type=float, default=0.01)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--out_dir", default="results/mulcp")
    parser.add_argument("--results_csv", default="")
    parser.add_argument("--log_progress_csv", nargs="?", const="auto", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dims = parse_dims(args.dims)
    dtype = get_dtype(args.dtype)
    budget = resolve_budget(dims, args.parameter_ratio, args.target_params)
    rank_left, rank_right = choose_rank_pair(
        dims, budget, args.rank_left, args.rank_right
    )

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
        model = MultiplicativeCP(
            dims,
            rank_left,
            rank_right,
            init_scale=args.init_scale,
            identity_right_init=args.identity_right_init,
            right_noise_std=args.right_noise_std,
            device=args.device,
            dtype=dtype,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        result = train_model(
            model,
            target,
            optimizer,
            epochs=args.epochs,
            method="mulcp",
            seed=seed,
            log_every=args.log_every,
            save_every=args.save_every,
            lr_schedule=args.lr_schedule,
            lr_floor=args.lr_floor,
            progress=progress,
        )
        row = make_result_row(
            seed=seed,
            method="mulcp",
            data=args.data,
            dims=dims,
            budget=budget,
            n_params=count_parameters(model),
            result=result,
            extras={"rank_left": rank_left, "rank_right": rank_right},
        )
        rows.append(row)
        save_rows(rows, results_path)
        print(
            f"[result] seed={seed} rel={result.rel_fro:.6e} "
            f"params={count_parameters(model)}/{budget}",
            flush=True,
        )
    print(f"Saved results to {os.path.abspath(results_path)}")


if __name__ == "__main__":
    main()
