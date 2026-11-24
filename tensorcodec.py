from __future__ import annotations

"""Integrated TensorCodec-style neural Tensor-Train baseline for synthetic tensors."""

import argparse
import math
import os
import time
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ProgressCSV,
    count_parameters,
    get_dtype,
    make_result_row,
    parse_dims,
    parse_int_list,
    prod,
    resolve_budget,
    save_rows,
    set_seed,
    TrainResult,
)
from synthetic import make_synthetic_tensor


def ceil_log2(value: int) -> int:
    value = max(1, int(value))
    return int(math.ceil(math.log2(value))) if value > 1 else 0


def power_two_factors(value: int, order: int) -> List[int]:
    active = min(int(order), ceil_log2(int(value)))
    return [1] * (int(order) - active) + [2] * active


def build_fold_factors(dims: Sequence[int]) -> List[List[int]]:
    folded_order = max(len(dims) + 1, max(2, ceil_log2(max(int(dim) for dim in dims))))
    return [power_two_factors(int(dim), folded_order) for dim in dims]


def folded_dims(factors: Sequence[Sequence[int]]) -> List[int]:
    order = len(factors[0])
    return [prod([row[column] for row in factors]) for column in range(order)]


@torch.no_grad()
def slice_mst_order(tensor: torch.Tensor, mode: int, max_features: int, seed: int) -> List[int]:
    work = tensor.detach().float().cpu()
    permutation = [mode] + [index for index in range(work.ndim) if index != mode]
    slices = work.permute(*permutation).contiguous().reshape(work.shape[mode], -1)
    count, features = slices.shape
    if count <= 2:
        return list(range(count))
    if features > int(max_features):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1009 * int(mode))
        indices = torch.randperm(features, generator=generator)[: int(max_features)]
        slices = slices.index_select(1, indices) * math.sqrt(features / float(max_features))
    distances = torch.cdist(slices, slices).numpy()

    in_tree = [False] * count
    key = [float("inf")] * count
    parent = [-1] * count
    key[0] = 0.0
    for _ in range(count):
        node = min((index for index in range(count) if not in_tree[index]), key=lambda index: key[index])
        in_tree[node] = True
        for candidate in range(count):
            distance = float(distances[node, candidate])
            if not in_tree[candidate] and distance < key[candidate]:
                key[candidate] = distance
                parent[candidate] = node

    adjacency: List[List[int]] = [[] for _ in range(count)]
    for node, ancestor in enumerate(parent):
        if ancestor >= 0:
            adjacency[node].append(ancestor)
            adjacency[ancestor].append(node)
    for neighbors in adjacency:
        neighbors.sort()

    order: List[int] = []
    stack = [0]
    visited = [False] * count
    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        order.append(node)
        for neighbor in reversed(adjacency[node]):
            if not visited[neighbor]:
                stack.append(neighbor)
    if len(order) > 2:
        edge_lengths = [float(distances[order[index], order[(index + 1) % count]]) for index in range(count)]
        cut = max(range(count), key=lambda index: edge_lengths[index])
        order = order[cut + 1 :] + order[: cut + 1]
    return order


def initialize_permutations(tensor: torch.Tensor, seed: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    permutations: List[torch.Tensor] = []
    inverses: List[torch.Tensor] = []
    for mode, dim in enumerate(tensor.shape):
        order = slice_mst_order(tensor, mode, max_features=4096, seed=seed)
        permutation = torch.tensor(order, dtype=torch.long, device=tensor.device)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(int(dim), device=tensor.device)
        permutations.append(permutation)
        inverses.append(inverse)
    return permutations, inverses


class IndexMapper:
    def __init__(
        self,
        dims: Sequence[int],
        fold_factors: Sequence[Sequence[int]],
        inverse_permutations: Sequence[torch.Tensor],
        device: str,
    ) -> None:
        self.dims = tuple(int(dim) for dim in dims)
        self.order = len(self.dims)
        self.fold_factors = [list(map(int, row)) for row in fold_factors]
        self.folded_order = len(self.fold_factors[0])
        self.device = torch.device(device)
        self.inverse_permutations = [value.to(self.device) for value in inverse_permutations]

        strides: List[int] = []
        running = 1
        for dim in reversed(self.dims):
            strides.insert(0, running)
            running *= dim
        self.strides = torch.tensor(strides, dtype=torch.long, device=self.device).view(1, self.order)
        self.dims_tensor = torch.tensor(self.dims, dtype=torch.long, device=self.device).view(1, self.order)

        fold = torch.tensor(self.fold_factors, dtype=torch.long, device=self.device)
        bases = []
        for row in self.fold_factors:
            row_bases = []
            running = 1
            for value in reversed(row):
                row_bases.insert(0, running)
                running *= int(value)
            bases.append(row_bases)
        self.fold = fold
        self.fold_bases = torch.tensor(bases, dtype=torch.long, device=self.device)

        column_multipliers: List[List[int]] = []
        for column in range(self.folded_order):
            multipliers = [1] * self.order
            running = 1
            for mode in range(self.order - 1, -1, -1):
                multipliers[mode] = running
                running *= self.fold_factors[mode][column]
            column_multipliers.append(multipliers)
        self.column_multipliers = torch.tensor(
            column_multipliers, dtype=torch.long, device=self.device
        ).transpose(0, 1)

        self.folded_dims = folded_dims(self.fold_factors)
        offsets: List[int] = []
        size_to_offset: Dict[int, int] = {}
        total_embeddings = 0
        for dimension in self.folded_dims:
            if dimension in size_to_offset:
                offsets.append(size_to_offset[dimension])
            else:
                offsets.append(total_embeddings)
                size_to_offset[dimension] = total_embeddings
                total_embeddings += dimension
        self.offsets = torch.tensor(offsets, dtype=torch.long, device=self.device).view(1, self.folded_order)
        self.num_embeddings = int(total_embeddings)

    def flat_to_input(self, flat_indices: torch.Tensor) -> torch.Tensor:
        flat_indices = flat_indices.to(self.device, dtype=torch.long).view(-1, 1)
        coordinates = (flat_indices // self.strides) % self.dims_tensor
        reordered = torch.stack(
            [
                self.inverse_permutations[mode].index_select(0, coordinates[:, mode])
                for mode in range(self.order)
            ],
            dim=1,
        )
        digits = (
            reordered.unsqueeze(-1) // self.fold_bases.unsqueeze(0)
        ) % self.fold.unsqueeze(0)
        folded = (digits * self.column_multipliers.unsqueeze(0)).sum(dim=1)
        return folded + self.offsets


class TensorCodecModel(nn.Module):
    def __init__(self, rank: int, hidden_size: int, num_embeddings: int, folded_order: int) -> None:
        super().__init__()
        self.rank = int(rank)
        self.hidden_size = int(hidden_size)
        self.folded_order = int(folded_order)
        self.embedding = nn.Embedding(int(num_embeddings), self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=False)
        self.first = nn.Linear(self.hidden_size, self.rank)
        self.middle = nn.Linear(self.hidden_size, self.rank * self.rank)
        self.last = nn.Linear(self.hidden_size, self.rank)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(model_input).transpose(0, 1).contiguous()
        self.lstm.flatten_parameters()
        hidden, _ = self.lstm(embedded)
        first = self.first(hidden[0]).unsqueeze(1)
        last = self.last(hidden[-1]).unsqueeze(-1)
        if self.folded_order <= 2:
            return torch.bmm(first, last).reshape(-1)
        middle = self.middle(hidden[1:-1]).view(
            self.folded_order - 2, -1, self.rank, self.rank
        )
        accumulator = first
        for index in range(self.folded_order - 2):
            accumulator = torch.bmm(accumulator, middle[index])
        return torch.bmm(accumulator, last).reshape(-1)


def model_parameter_count(rank: int, hidden: int, mapper: IndexMapper) -> int:
    model = TensorCodecModel(rank, hidden, mapper.num_embeddings, mapper.folded_order)
    return count_parameters(model)


def choose_hyperparameters(
    mapper: IndexMapper,
    budget: int,
    rank_override: int,
    hidden_override: int,
) -> Tuple[int, int]:
    ranks = [int(rank_override)] if int(rank_override) > 0 else list(range(1, 65))
    hidden_sizes = [int(hidden_override)] if int(hidden_override) > 0 else list(range(2, 81))
    best: Tuple[int, int, int] | None = None
    for rank in ranks:
        for hidden in hidden_sizes:
            parameters = model_parameter_count(rank, hidden, mapper)
            if parameters <= int(budget) and (best is None or parameters > best[0]):
                best = (parameters, rank, hidden)
    if best is None:
        raise ValueError("No TensorCodec rank-hidden pair fits the target budget")
    return int(best[1]), int(best[2])


@torch.no_grad()
def full_metrics(
    model: TensorCodecModel,
    mapper: IndexMapper,
    values: torch.Tensor,
    batch_size: int,
) -> Tuple[float, float]:
    model.eval()
    squared_error = 0.0
    squared_norm = 0.0
    count = values.numel()
    for start in range(0, count, int(batch_size)):
        end = min(start + int(batch_size), count)
        indices = torch.arange(start, end, device=values.device)
        prediction = model(mapper.flat_to_input(indices))
        target = values.index_select(0, indices)
        residual = prediction - target
        squared_error += float(residual.square().sum().detach().cpu().item())
        squared_norm += float(target.square().sum().detach().cpu().item())
    return squared_error / count, math.sqrt(squared_error) / max(math.sqrt(squared_norm), 1e-12)


def train_tensorcodec(
    target: torch.Tensor,
    model: TensorCodecModel,
    mapper: IndexMapper,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    eval_batch_size: int,
    log_every: int,
    seed: int,
    progress: ProgressCSV,
) -> TrainResult:
    values = target.reshape(-1)
    count = values.numel()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best_mse = float("inf")
    best_rel = float("inf")
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    start_time = time.time()

    for epoch in range(int(epochs)):
        model.train()
        permutation = torch.randperm(count, device=target.device)
        for start in range(0, count, int(batch_size)):
            indices = permutation[start : start + int(batch_size)]
            prediction = model(mapper.flat_to_input(indices))
            batch_target = values.index_select(0, indices)
            loss = F.mse_loss(prediction, batch_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        one_based = epoch + 1
        if one_based == 1 or one_based == int(epochs) or one_based % int(log_every) == 0:
            mse, rel = full_metrics(model, mapper, values, eval_batch_size)
            if rel < best_rel:
                best_mse = mse
                best_rel = rel
                best_epoch = one_based
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            elapsed = time.time() - start_time
            print(
                f"[tensorcodec] epoch={one_based:05d}/{epochs} rel={rel:.6e} "
                f"best={best_rel:.6e}@{best_epoch} time={elapsed:.2f}s",
                flush=True,
            )
            progress.append(
                {
                    "seed": seed,
                    "method": "tensorcodec",
                    "epoch": one_based,
                    "total_epochs": epochs,
                    "elapsed_total_sec": elapsed,
                    "current_mse": mse,
                    "current_rel_fro": rel,
                    "best_rel_fro": best_rel,
                    "best_epoch": best_epoch,
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_mse, _ = full_metrics(model, mapper, values, eval_batch_size)
    return TrainResult(
        best_epoch=best_epoch,
        train_time_sec=time.time() - start_time,
        mse=best_mse,
        rel_fro=best_rel,
        final_mse=final_mse,
    )


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
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--hidden_size", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--eval_batch_size", type=int, default=65536)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--out_dir", default="results/tensorcodec")
    parser.add_argument("--results_csv", default="")
    parser.add_argument("--log_progress_csv", nargs="?", const="auto", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.log_every <= 0:
        raise ValueError("log_every must be positive")
    dims = parse_dims(args.dims)
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
        _, inverses = initialize_permutations(target, seed)
        factors = build_fold_factors(dims)
        mapper = IndexMapper(dims, factors, inverses, args.device)
        rank, hidden = choose_hyperparameters(mapper, budget, args.rank, args.hidden_size)
        model = TensorCodecModel(rank, hidden, mapper.num_embeddings, mapper.folded_order).to(
            device=args.device, dtype=dtype
        )
        result = train_tensorcodec(
            target,
            model,
            mapper,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            log_every=args.log_every,
            seed=seed,
            progress=progress,
        )
        row = make_result_row(
            seed=seed,
            method="tensorcodec",
            data=args.data,
            dims=dims,
            budget=budget,
            n_params=count_parameters(model),
            result=result,
            extras={
                "rank": rank,
                "hidden_size": hidden,
                "folded_order": mapper.folded_order,
                "folded_dims": "x".join(map(str, mapper.folded_dims)),
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
