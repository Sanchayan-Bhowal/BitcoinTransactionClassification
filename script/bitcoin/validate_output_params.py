"""Validate saved MCEM beta/gamma parameters from output.txt.

This script recovers the final beta/gamma from a printed `history = [...]`
block, recreates a stratified held-out split over known Elliptic labels, and
scores the held-out nodes with the fitted Ising conditional field.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.datasets import EllipticBitcoinDataset


def parse_final_params(output_path: Path) -> tuple[float, np.ndarray, dict]:
    text = output_path.read_text(encoding="utf-8")
    match = re.search(r"history\s*=\s*(\[.*\])\s*$", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not find a `history = [...]` block in {output_path}.")

    history = eval(match.group(1), {"__builtins__": {}}, {"array": np.array})
    final = history[-1]
    return float(final["beta"]), np.asarray(final["gamma"], dtype=float), final


def build_symmetric_adjacency(num_nodes: int, edge_index: torch.Tensor) -> sp.csr_matrix:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    keep = src != dst
    src = src[keep]
    dst = dst[keep]

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    values = np.ones(rows.shape[0], dtype=np.float32)
    adjacency = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    return adjacency


def normalize_elliptic_labels(y_raw: torch.Tensor) -> torch.Tensor:
    """Map this repo's labels to Ising spins: licit=0 -> -1, illicit=1 -> +1, unknown=2 -> 0."""
    y_pm1 = torch.zeros(y_raw.numel(), dtype=torch.int8)
    y_pm1[y_raw == 0] = -1
    y_pm1[y_raw == 1] = 1
    y_pm1[y_raw == 2] = 0
    return y_pm1


def make_stratified_test_mask(y_pm1: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
    test_mask = torch.zeros(y_pm1.numel(), dtype=torch.bool)
    generator = torch.Generator().manual_seed(seed)

    for label in (-1, 1):
        class_idx = torch.nonzero(y_pm1 == label, as_tuple=False).flatten()
        if class_idx.numel() <= 1:
            continue

        num_test = int(round(fraction * class_idx.numel()))
        num_test = max(1, num_test)
        num_test = min(num_test, class_idx.numel() - 1)
        shuffled = class_idx[torch.randperm(class_idx.numel(), generator=generator)]
        test_mask[shuffled[:num_test]] = True

    return test_mask


def standardize_features(X: torch.Tensor) -> np.ndarray:
    X = X.float()
    mean = X.mean(dim=0, keepdim=True)
    scale = X.std(dim=0, unbiased=False, keepdim=True)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return ((X - mean) / scale).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output.txt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/EllipticBitcoinDataset"))
    parser.add_argument("--num-features", type=int, default=94)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=305)
    parser.add_argument("--metrics-out", type=Path, default=Path("figures/test_metrics_from_output_params.txt"))
    args = parser.parse_args()

    beta, gamma, final = parse_final_params(args.output)
    if gamma.shape[0] != args.num_features:
        raise ValueError(f"gamma has length {gamma.shape[0]}, expected {args.num_features}.")

    dataset = EllipticBitcoinDataset(root=str(args.data_root))
    data = dataset[0]
    X = data.x[:, : args.num_features].float().cpu()
    y_pm1 = normalize_elliptic_labels(data.y.long().cpu())
    adjacency = build_symmetric_adjacency(num_nodes=X.shape[0], edge_index=data.edge_index.long().cpu())

    test_mask = make_stratified_test_mask(y_pm1, fraction=args.test_fraction, seed=args.seed)
    y_train = y_pm1.clone()
    y_train[test_mask] = 0

    X_std = standardize_features(X)
    sigma_context = y_train.cpu().numpy().astype(np.float64)
    neighbor_sum = np.asarray(adjacency @ sigma_context, dtype=np.float64).reshape(-1)

    # Plug-in conditional probability with training labels clamped. Held-out
    # known labels and unknown labels are hidden as 0 in sigma_context.
    field = 2.0 * (beta * neighbor_sum + X_std @ gamma)
    score_all = 1.0 / (1.0 + np.exp(-field))
    pred_spin_all = np.where(score_all >= 0.5, 1, -1).astype(np.int8)

    test_np = test_mask.cpu().numpy()
    y_true = (y_pm1.cpu().numpy()[test_np] == 1).astype(int)
    y_pred = (pred_spin_all[test_np] == 1).astype(int)
    y_score = score_all[test_np]

    metrics = {
        "source_iteration": int(final["iteration"]),
        "num_nodes": int(y_pm1.numel()),
        "num_undirected_edges": int(adjacency.nnz // 2),
        "num_test": int(y_true.size),
        "num_test_licit": int((y_true == 0).sum()),
        "num_test_illicit": int((y_true == 1).sum()),
        "beta": beta,
        "gamma_l2_norm": float(np.linalg.norm(gamma)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_illicit": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_illicit": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_illicit": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "confusion_matrix_licit_illicit": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "score_note": (
            "Plug-in conditional probability with training labels clamped and "
            "held-out/unknown labels hidden; not a full Gibbs posterior rerun."
        ),
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
