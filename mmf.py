from __future__ import annotations

"""Fixed-mask mixture factorization baseline on synthetic tensors."""

import argparse
import math
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


def fixed_profile(query: torch.Tensor, family: str, width: float) -> torch.Tensor:
    family = str(family).lower()
    centered = torch.remainder(query + 0.5, 1.0) - 0.5
    if family == "sin":
        return torch.sin(2.0 * math.pi * query)
    if family == "cos":
        return torch.cos(2.0 * math.pi * query)
    if family == "sigmoid":
        return torch.sigmoid(centered / max(float(width), 1e-8))
    if family == "gaussian":
        return torch.exp(-0.5 * (centered / max(float(width), 1e-8)).square())
    if family == "triangle":
        return torch.clamp(1.0 - centered.abs() / max(float(width), 1e-8), min=0.0)
    if family == "tanh":
        return torch.tanh(centered / max(float(width), 1e-8))
    raise ValueError(f"Unknown fixed mask family: {family}")


class FixedMMF(DenseTensorModel):
    def __init__(
        self,
        dims: Sequence[int],
        rank: int,
        components: int,
        *,
        mask_family: str,
        mask_width: float,
        learnable_omega: bool,
        init_scale: float,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.dims = tuple(int(dim) for dim in dims)
        self.rank = int(rank)
        self.components = int(components)
        self.mask_family = str(mask_family)
        self.mask_width = float(mask_width)
        self.learnable_omega = bool(learnable_omega)

        self.factors = nn.ParameterList(
            [
                nn.Parameter(float(init_scale) * torch.randn(dim, self.rank, device=device, dtype=dtype))
                for dim in self.dims
            ]
        )
        self.shifts = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim, self.components, device=device, dtype=dtype)) for dim in self.dims]
        )
        position = torch.arange(self.rank, device=device, dtype=dtype) / float(self.rank)
        self.register_buffer("position", position.view(1, 1, self.rank), persistent=False)
        omega = torch.linspace(
            1.0 / self.components,
            1.0,
            steps=self.components,
            device=device,
            dtype=dtype,
        ).view(1, self.components, 1)
        if self.learnable_omega:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega, persistent=True)

    def effective_factors(self) -> List[torch.Tensor]:
        output: List[torch.Tensor] = []
        for dim, factor, shift in zip(self.dims, self.factors, self.shifts):
            query = torch.remainder(self.position * self.omega - shift.unsqueeze(-1), 1.0)
            mask = fixed_profile(query, self.mask_family, self.mask_width)
            output.append((factor.unsqueeze(1) * mask).reshape(dim, self.components * self.rank))
        return output

    def cp_factors(self) -> List[torch.Tensor]:
        return self.effective_factors()

    def reconstruct(self) -> torch.Tensor:
        return cp_to_tensor(self.effective_factors())


def choose_components(
    dims: Sequence[int], rank: int, budget: int, learnable_omega: bool
) -> int:
    total_mode_size = sum(int(dim) for dim in dims)
    fixed = int(rank) * total_mode_size
    per_component = total_mode_size + (1 if learnable_omega else 0)
    remaining = int(budget) - fixed
    if remaining < per_component:
        raise ValueError("The budget is too small for the requested MMF rank")
    return max(1, remaining // per_component)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", default="64,64,64")
    parser.add_argument("--data", choices=["random", "block"], default="random")
    parser.add_argument("--distribution", choices=["normal", "uniform"], default="normal")
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--block_scale", type=float, default=1.0)
    parser.add_argument("--off_block_scale", type=float, default=0.0)

    parser.add_argument("--parameter_ratio", type=float, default=0.125)
    parser.add_argument("--target_params", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--components", type=int, default=-1)
    parser.add_argument("--mask_family", choices=["sin", "cos", "sigmoid", "gaussian", "triangle", "tanh"], default="sin")
    parser.add_argument("--mask_width", type=float, default=1.0)
    parser.add_argument("--learnable_omega", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--out_dir", default="results/mmf")
    parser.add_argument("--results_csv", default="")
    parser.add_argument("--log_progress_csv", nargs="?", const="auto", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dims = parse_dims(args.dims)
    dtype = get_dtype(args.dtype)
    budget = resolve_budget(dims, args.parameter_ratio, args.target_params)
    components = (
        int(args.components)
        if int(args.components) > 0
        else choose_components(dims, args.rank, budget, args.learnable_omega)
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
            components=components,
            distribution=args.distribution,
            num_blocks=args.num_blocks,
            block_scale=args.block_scale,
            off_block_scale=args.off_block_scale,
            device=args.device,
            dtype=dtype,
        )
        model = FixedMMF(
            dims,
            args.rank,
            components,
            mask_family=args.mask_family,
            mask_width=args.mask_width,
            learnable_omega=args.learnable_omega,
            init_scale=args.init_scale,
            device=args.device,
            dtype=dtype,
        )
        if count_parameters(model) > budget:
            raise ValueError(
                f"MMF configuration uses {count_parameters(model)} parameters, exceeding budget {budget}"
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        result = train_model(
            model,
            target,
            optimizer,
            epochs=args.epochs,
            method="mmf",
            seed=seed,
            log_every=args.log_every,
            save_every=args.save_every,
            lr_schedule=args.lr_schedule,
            lr_floor=args.lr_floor,
            progress=progress,
        )
        row = make_result_row(
            seed=seed,
            method="mmf",
            data=args.data,
            dims=dims,
            budget=budget,
            n_params=count_parameters(model),
            result=result,
            extras={
                "rank": args.rank,
                "components": components,
                "mask_family": args.mask_family,
                "mask_width": args.mask_width,
            },
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
