from __future__ import annotations

"""Synthetic comparison of MorphLaM with CP, Tucker, and Tensor-Train."""

import argparse
import math
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from classical import (
    CPModel,
    TTModel,
    TuckerModel,
    cp_als,
    hooi,
    tt_svd_permutation_search,
)
from common import (
    DenseTensorModel,
    ProgressCSV,
    choose_cp_rank,
    choose_morphlam_components,
    choose_tt_ranks,
    choose_tucker_ranks,
    count_parameters,
    cp_parameter_count,
    cp_to_tensor,
    get_dtype,
    make_result_row,
    morphlam_parameter_count,
    parse_dims,
    parse_int_list,
    parse_str_list,
    resolve_budget,
    save_rows,
    set_seed,
    train_model,
    tt_parameter_count,
    tucker_parameter_count,
)
from synthetic import make_synthetic_tensor


class PoissonLookupProfile(nn.Module):
    def __init__(
        self,
        num_basis: int,
        lookup_size: int,
        rho_init: float,
        rho_min: float,
        rho_max: float,
        *,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_basis = int(num_basis)
        self.lookup_size = int(lookup_size)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        if self.num_basis < 1 or self.lookup_size < 2:
            raise ValueError("num_basis must be positive and lookup_size must be at least two")
        if not (0.0 <= self.rho_min < float(rho_init) < self.rho_max < 1.0):
            raise ValueError("Need 0 <= rho_min < rho_init < rho_max < 1")

        grid = torch.arange(self.lookup_size, device=device, dtype=dtype) / float(self.lookup_size)
        centers = torch.arange(self.num_basis, device=device, dtype=dtype) / float(self.num_basis)
        cos_delta = 2.0 * torch.cos(2.0 * math.pi * (grid[:, None] - centers[None, :]))
        self.register_buffer("cos_delta", cos_delta.contiguous(), persistent=False)
        self.coeff = nn.Parameter(torch.ones(self.num_basis, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.rho_raw = nn.Parameter(self._rho_to_raw(torch.tensor(float(rho_init), device=device, dtype=dtype)))
        self.rho_update_clip = 0.0
        self.rho_floor: Optional[float] = None
        self._rho_before: Optional[torch.Tensor] = None

    def _rho_to_raw(self, rho: torch.Tensor) -> torch.Tensor:
        normalized = (rho - self.rho_min) / (self.rho_max - self.rho_min)
        return torch.logit(normalized.clamp(1e-7, 1.0 - 1e-7))

    def rho(self) -> torch.Tensor:
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(self.rho_raw)

    def set_rho_update_clip(self, value: float) -> None:
        self.rho_update_clip = max(0.0, float(value))

    def set_rho_floor(self, value: Optional[float]) -> None:
        self.rho_floor = None if value is None else float(value)

    def build_grid(self) -> torch.Tensor:
        rho = self.rho().reshape(())
        if self.rho_update_clip > 0.0:
            self._rho_before = rho.detach()
        rho_squared = rho * rho
        denominator = 1.0 - rho * self.cos_delta + rho_squared
        basis = (1.0 / self.num_basis) * (1.0 - rho_squared) / denominator
        return torch.addmv(self.bias.expand(self.lookup_size), basis, self.coeff)

    def sample(self, grid_values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        position = torch.remainder(query, 1.0) * float(self.lookup_size)
        left = torch.floor(position).long() % self.lookup_size
        alpha = position - torch.floor(position)
        right = (left + 1) % self.lookup_size
        return (1.0 - alpha) * grid_values[left] + alpha * grid_values[right]

    @torch.no_grad()
    def apply_constraints_(self) -> None:
        if self.rho_update_clip <= 0.0 and self.rho_floor is None:
            return
        current = self.rho()
        lower = torch.tensor(self.rho_min, device=current.device, dtype=current.dtype)
        upper = torch.tensor(self.rho_max, device=current.device, dtype=current.dtype)
        if self.rho_floor is not None:
            lower = torch.maximum(
                lower,
                torch.tensor(self.rho_floor, device=current.device, dtype=current.dtype),
            )
        if self.rho_update_clip > 0.0 and self._rho_before is not None:
            lower = torch.maximum(lower, self._rho_before - self.rho_update_clip)
            upper = torch.minimum(upper, self._rho_before + self.rho_update_clip)
            if self.rho_floor is not None:
                upper = torch.maximum(
                    upper,
                    torch.tensor(self.rho_floor, device=current.device, dtype=current.dtype),
                )
        upper = torch.maximum(upper, lower)
        clipped = torch.minimum(torch.maximum(current, lower), upper)
        self.rho_raw.copy_(self._rho_to_raw(clipped))
        self._rho_before = None


class MorphLaM(DenseTensorModel):
    def __init__(
        self,
        dims: Sequence[int],
        rank: int,
        components: int,
        num_basis: int,
        lookup_size: int,
        rho_init: float,
        rho_min: float,
        rho_max: float,
        *,
        shared_profile: bool,
        shared_omega: bool,
        shift_init_std: float,
        init_scale: float,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.dims = tuple(int(dim) for dim in dims)
        self.rank = int(rank)
        self.components = int(components)
        self.shared_profile = bool(shared_profile)
        self.shared_omega = bool(shared_omega)
        if self.rank < 1 or self.components < 1:
            raise ValueError("rank and components must be positive")

        self.factors = nn.ParameterList(
            [
                nn.Parameter(float(init_scale) * torch.randn(dim, self.rank, device=device, dtype=dtype))
                for dim in self.dims
            ]
        )
        self.shifts = nn.ParameterList(
            [
                nn.Parameter(float(shift_init_std) * torch.randn(dim, self.components, device=device, dtype=dtype))
                for dim in self.dims
            ]
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
        if self.shared_omega:
            self.omega = nn.Parameter(omega.clone())
            self.omegas = None
        else:
            self.omega = None
            self.omegas = nn.ParameterList([nn.Parameter(omega.clone()) for _ in self.dims])

        def new_profile() -> PoissonLookupProfile:
            return PoissonLookupProfile(
                num_basis,
                lookup_size,
                rho_init,
                rho_min,
                rho_max,
                device=device,
                dtype=dtype,
            )

        if self.shared_profile:
            self.profile = new_profile()
            self.profiles = None
        else:
            self.profile = None
            self.profiles = nn.ModuleList([new_profile() for _ in self.dims])

    def profile_modules(self) -> List[PoissonLookupProfile]:
        if self.shared_profile:
            assert self.profile is not None
            return [self.profile]
        assert self.profiles is not None
        return list(self.profiles)

    def omega_for_mode(self, mode: int) -> torch.Tensor:
        if self.shared_omega:
            assert self.omega is not None
            return self.omega
        assert self.omegas is not None
        return self.omegas[mode]

    def set_rho_update_clip(self, value: float) -> None:
        for profile in self.profile_modules():
            profile.set_rho_update_clip(value)

    def set_rho_floor(self, value: Optional[float]) -> None:
        for profile in self.profile_modules():
            profile.set_rho_floor(value)

    @torch.no_grad()
    def apply_post_step_constraints_(self) -> None:
        for profile in self.profile_modules():
            profile.apply_constraints_()

    def effective_factors(self) -> List[torch.Tensor]:
        queries = [
            torch.remainder(
                self.position * self.omega_for_mode(mode) - self.shifts[mode].unsqueeze(-1),
                1.0,
            )
            for mode in range(len(self.dims))
        ]
        masks: List[torch.Tensor] = []
        if self.shared_profile:
            assert self.profile is not None
            grid = self.profile.build_grid()
            masks = [self.profile.sample(grid, query) for query in queries]
        else:
            assert self.profiles is not None
            for profile, query in zip(self.profiles, queries):
                masks.append(profile.sample(profile.build_grid(), query))
        return [
            (factor.unsqueeze(1) * mask).reshape(dim, self.components * self.rank)
            for dim, factor, mask in zip(self.dims, self.factors, masks)
        ]

    def cp_factors(self) -> List[torch.Tensor]:
        return self.effective_factors()

    def reconstruct(self) -> torch.Tensor:
        return cp_to_tensor(self.effective_factors())


def morphlam_optimizer(model: MorphLaM, factor_lr: float, profile_lr: float) -> torch.optim.Optimizer:
    factor_side: List[nn.Parameter] = [*model.factors, *model.shifts]
    if model.shared_omega:
        assert model.omega is not None
        factor_side.append(model.omega)
    else:
        assert model.omegas is not None
        factor_side.extend(model.omegas)
    profile_side: List[nn.Parameter] = []
    for profile in model.profile_modules():
        profile_side.extend(profile.parameters())
    return torch.optim.Adam(
        [
            {"params": factor_side, "lr": float(factor_lr)},
            {"params": profile_side, "lr": float(profile_lr)},
        ]
    )


def parse_rank_vector(value: str, expected: int) -> Optional[List[int]]:
    values = parse_int_list(value)
    if not values:
        return None
    if len(values) == 1:
        return values * expected
    if len(values) != expected:
        raise ValueError(f"Expected one rank or {expected} ranks; got {values}")
    return values


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
    parser.add_argument("--rank", type=int, default=24, help="MorphLaM base rank R")
    parser.add_argument("--components", type=int, default=-1, help="MorphLaM components K; <=0 fills the budget")
    parser.add_argument(
        "--methods",
        default="morphlam,cp,cp_als,tucker,tucker_hooi,tt,tt_svd_perm",
    )

    parser.add_argument("--num_basis", type=int, default=1024)
    parser.add_argument("--lookup_size", type=int, default=4096)
    parser.add_argument("--rho_init", type=float, default=0.67)
    parser.add_argument("--rho_min", type=float, default=1e-4)
    parser.add_argument("--rho_max", type=float, default=0.9999)
    parser.add_argument("--rho_update_clip", type=float, default=2e-4)
    parser.add_argument("--rho_lower_schedule", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shift_init_std", type=float, default=1e-2)
    parser.add_argument("--shared_profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shared_omega", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init_scale", type=float, default=0.02)

    parser.add_argument("--cp_rank", type=int, default=-1)
    parser.add_argument("--tucker_ranks", default="")
    parser.add_argument("--tt_ranks", default="")
    parser.add_argument("--tt_permutations", default="auto")

    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--factor_lr", type=float, default=0.2)
    parser.add_argument("--profile_lr", type=float, default=0.1)
    parser.add_argument("--baseline_lr", type=float, default=0.01)
    parser.add_argument("--lr_schedule", choices=["none", "cosine_floor"], default="cosine_floor")
    parser.add_argument("--lr_floor", type=float, default=0.001)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=1000)

    parser.add_argument("--cp_als_sweeps", type=int, default=500)
    parser.add_argument("--cp_als_restarts", type=int, default=3)
    parser.add_argument("--hooi_sweeps", type=int, default=100)
    parser.add_argument("--hooi_restarts", type=int, default=2)
    parser.add_argument("--solver_tolerance", type=float, default=1e-7)
    parser.add_argument("--solver_verbose", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="highest")
    parser.add_argument("--out_dir", default="results/morphlam")
    parser.add_argument("--results_csv", default="")
    parser.add_argument(
        "--log_progress_csv",
        nargs="?",
        const="auto",
        default="",
        help="Write training checkpoints to CSV; optionally provide a path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dims = parse_dims(args.dims)
    methods = parse_str_list(args.methods)
    supported = {"morphlam", "cp", "cp_als", "tucker", "tucker_hooi", "tt", "tt_svd_perm"}
    unknown = sorted(set(methods) - supported)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.set_float32_matmul_precision(args.matmul_precision)
    dtype = get_dtype(args.dtype)
    budget = resolve_budget(dims, args.parameter_ratio, args.target_params)
    components = (
        int(args.components)
        if int(args.components) > 0
        else choose_morphlam_components(
            dims,
            args.rank,
            budget,
            args.num_basis,
            args.shared_profile,
            args.shared_omega,
        )
    )
    morphlam_params = morphlam_parameter_count(
        dims,
        args.rank,
        components,
        args.num_basis,
        args.shared_profile,
        args.shared_omega,
    )
    if morphlam_params > budget:
        raise ValueError(
            f"MorphLaM configuration uses {morphlam_params} parameters, exceeding budget {budget}"
        )

    cp_rank = int(args.cp_rank) if int(args.cp_rank) > 0 else choose_cp_rank(dims, budget)
    tucker_ranks = parse_rank_vector(args.tucker_ranks, len(dims)) or choose_tucker_ranks(dims, budget)
    tt_ranks = parse_rank_vector(args.tt_ranks, len(dims) - 1) or choose_tt_ranks(dims, budget)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = args.results_csv or os.path.join(args.out_dir, "results.csv")
    if args.log_progress_csv == "auto":
        progress_path = os.path.join(args.out_dir, "progress.csv")
    else:
        progress_path = args.log_progress_csv
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
        for method in methods:
            set_seed(seed, args.deterministic)
            if method == "morphlam":
                model = MorphLaM(
                    dims,
                    args.rank,
                    components,
                    args.num_basis,
                    args.lookup_size,
                    args.rho_init,
                    args.rho_min,
                    args.rho_max,
                    shared_profile=args.shared_profile,
                    shared_omega=args.shared_omega,
                    shift_init_std=args.shift_init_std,
                    init_scale=args.init_scale,
                    device=args.device,
                    dtype=dtype,
                )
                model.set_rho_update_clip(args.rho_update_clip)
                optimizer = morphlam_optimizer(model, args.factor_lr, args.profile_lr)

                def update_rho_floor(epoch: int, total_epochs: int) -> None:
                    if not args.rho_lower_schedule:
                        model.set_rho_floor(None)
                        return
                    progress_value = epoch / float(max(total_epochs - 1, 1))
                    floor = args.rho_init + (0.99 - args.rho_init) * math.sqrt(progress_value)
                    model.set_rho_floor(floor)

                result = train_model(
                    model,
                    target,
                    optimizer,
                    epochs=args.epochs,
                    method=method,
                    seed=seed,
                    log_every=args.log_every,
                    save_every=args.save_every,
                    lr_schedule=args.lr_schedule,
                    lr_floor=args.lr_floor,
                    progress=progress,
                    pre_epoch=update_rho_floor,
                )
                row = make_result_row(
                    seed=seed,
                    method=method,
                    data=args.data,
                    dims=dims,
                    budget=budget,
                    n_params=count_parameters(model),
                    result=result,
                    extras={"rank": args.rank, "components": components},
                )

            elif method == "cp":
                model = CPModel(
                    dims,
                    cp_rank,
                    init_scale=args.init_scale,
                    device=args.device,
                    dtype=dtype,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=args.baseline_lr)
                result = train_model(
                    model,
                    target,
                    optimizer,
                    epochs=args.epochs,
                    method=method,
                    seed=seed,
                    log_every=args.log_every,
                    save_every=args.save_every,
                    lr_schedule=args.lr_schedule,
                    lr_floor=args.lr_floor,
                    progress=progress,
                )
                row = make_result_row(
                    seed=seed,
                    method=method,
                    data=args.data,
                    dims=dims,
                    budget=budget,
                    n_params=cp_parameter_count(dims, cp_rank),
                    result=result,
                    extras={"rank": cp_rank},
                )

            elif method == "cp_als":
                solved = cp_als(
                    target,
                    cp_rank,
                    sweeps=args.cp_als_sweeps,
                    tolerance=args.solver_tolerance,
                    restarts=args.cp_als_restarts,
                    init_scale=args.init_scale,
                    seed=seed,
                    verbose=args.solver_verbose,
                )
                row = make_result_row(
                    seed=seed,
                    method=method,
                    data=args.data,
                    dims=dims,
                    budget=budget,
                    n_params=cp_parameter_count(dims, cp_rank),
                    result=solved.train,
                    extras={"rank": cp_rank, "restart": solved.restart},
                )

            elif method == "tucker":
                model = TuckerModel(
                    dims,
                    tucker_ranks,
                    init_scale=args.init_scale,
                    device=args.device,
                    dtype=dtype,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=args.baseline_lr)
                result = train_model(
                    model,
                    target,
                    optimizer,
                    epochs=args.epochs,
                    method=method,
                    seed=seed,
                    log_every=args.log_every,
                    save_every=args.save_every,
                    lr_schedule=args.lr_schedule,
                    lr_floor=args.lr_floor,
                    progress=progress,
                )
                row = make_result_row(
                    seed=seed,
                    method=method,
                    data=args.data,
                    dims=dims,
                    budget=budget,
                    n_params=tucker_parameter_count(dims, tucker_ranks),
                    result=result,
                    extras={"ranks": "x".join(map(str, tucker_ranks))},
                )

            elif method == "tucker_hooi":
                solved = hooi(
                    target,
                    tucker_ranks,
                    sweeps=args.hooi_sweeps,
                    tolerance=args.solver_tolerance,
                    restarts=args.hooi_restarts,
                    seed=seed,
                    verbose=args.solver_verbose,
                )
                row = make_result_row(
                    seed=seed,
                    method=method,
                    data=args.data,
                    dims=dims,
                    budget=budget,
                    n_params=tucker_parameter_count(dims, tucker_ranks),
                    result=solved.train,
                    extras={"ranks": "x".join(map(str, tucker_ranks)), "restart": solved.restart},
                )

            elif method == "tt":
                model = TTModel(
                    dims,
                    tt_ranks,
                    init_scale=args.init_scale,
                    device=args.device,
                    dtype=dtype,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=args.baseline_lr)
                result = train_model(
                    model,
                    target,
                    optimizer,
                    epochs=args.epochs,
                    method=method,
                    seed=seed,
                    log_every=args.log_every,
                    save_every=args.save_every,
                    lr_schedule=args.lr_schedule,
                    lr_floor=args.lr_floor,
                    progress=progress,
                )
                row = make_result_row(
                    seed=seed,
                    method=method,
                    data=args.data,
                    dims=dims,
                    budget=budget,
                    n_params=tt_parameter_count(dims, tt_ranks),
                    result=result,
                    extras={"ranks": "x".join(map(str, tt_ranks)), "permutation": "natural"},
                )

            else:
                fixed = parse_rank_vector(args.tt_ranks, len(dims) - 1)
                solved = tt_svd_permutation_search(
                    target,
                    budget,
                    permutations_spec=args.tt_permutations,
                    fixed_ranks=fixed,
                    verbose=args.solver_verbose,
                )
                permuted_dims = [dims[index] for index in solved.permutation]
                row = make_result_row(
                    seed=seed,
                    method=method,
                    data=args.data,
                    dims=dims,
                    budget=budget,
                    n_params=tt_parameter_count(permuted_dims, solved.ranks),
                    result=solved.train,
                    extras={
                        "ranks": "x".join(map(str, solved.ranks)),
                        "permutation": ",".join(map(str, solved.permutation)),
                    },
                )

            rows.append(row)
            save_rows(rows, results_path)
            print(
                f"[result] method={row['method']} seed={seed} rel={float(row['rel_fro']):.6e} "
                f"params={row['n_params']}/{budget}",
                flush=True,
            )

    print(f"Saved results to {os.path.abspath(results_path)}")


if __name__ == "__main__":
    main()
