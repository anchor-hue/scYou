#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spectral eigengap cluster number estimation.

Reference: von Luxburg (2007), "A tutorial on spectral clustering",
Statistics and Computing, 17(4), 395-416.

Method: build a symmetric KNN graph from PCA-reduced data, compute the
normalized Laplacian, and find K where the eigenvalue gap λ_{K+1} - λ_K
is largest. This is the standard perturbation-theoretic criterion for
the number of connected components in spectral clustering.
"""

from __future__ import annotations

import argparse, math, zlib, sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def parse_args():
    p = argparse.ArgumentParser(description="Spectral eigengap estimation.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pattern", type=str, default="*.csv")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def dataset_name(path: Path) -> str:
    name = path.stem
    for prefix in ("expression_", "processed_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def read_matrix(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, index_col=0)
    if raw.index.duplicated().any():
        raw = raw.groupby(level=0, sort=False).mean(numeric_only=False)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    numeric = numeric.fillna(0.0)
    matrix = numeric.T  # protein×cell → cell×protein
    variance = matrix.var(axis=0, ddof=0)
    matrix = matrix.loc[:, variance > 0]
    if matrix.shape[0] < 4:
        raise ValueError(f"{path.name}: fewer than four cells.")
    return matrix


def build_pca(matrix: pd.DataFrame, n_components: int, seed: int):
    x = matrix.to_numpy(dtype=np.float32, copy=True)
    dim = min(n_components, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(x)
    return z, dim, float(pca.explained_variance_ratio_.sum())


def estimate_eigengap(z: np.ndarray, k_min: int, k_max: int):
    """Spectral eigengap: argmax λ_{K+1} - λ_K on normalized Laplacian."""
    n_cells = z.shape[0]

    # Adaptive neighborhood size
    n_neighbors = min(30, max(15, int(math.floor(math.sqrt(n_cells)))), n_cells - 1)

    # Symmetric KNN graph
    graph = kneighbors_graph(z, n_neighbors=n_neighbors, mode="connectivity",
                             include_self=False, n_jobs=-1)
    graph = graph.maximum(graph.T)

    # Normalized Laplacian eigenvalues
    L = laplacian(graph, normed=True)
    n_eig = min(k_max + 1, n_cells - 2)
    eigenvalues = eigsh(L, k=n_eig, which="SM", return_eigenvectors=False, tol=1e-3)
    eigenvalues = np.sort(np.real(eigenvalues))

    # Eigengap: λ_{K+1} - λ_K
    gaps = {}
    upper = min(k_max, len(eigenvalues) - 1)
    for k in range(k_min, upper + 1):
        gaps[k] = float(eigenvalues[k] - eigenvalues[k - 1])

    k_est = max(gaps, key=gaps.get)
    return k_est, n_neighbors, eigenvalues, gaps


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.input_dir.glob(args.pattern) if p.is_file())
    if not files:
        print("No CSV files found.", file=sys.stderr)
        return 1

    rows = []
    for path in files:
        name = dataset_name(path)
        matrix = read_matrix(path)
        seed = (args.seed + int(zlib.crc32(name.encode("utf-8")))) % (2**32 - 1)

        z, pca_dim, pca_var = build_pca(matrix, args.n_components, seed)
        k_max = min(args.k_max, z.shape[0] - 1)

        k_est, n_neighbors, eigenvalues, gaps = estimate_eigengap(z, args.k_min, k_max)

        rows.append({
            "dataset": name,
            "n_cells": matrix.shape[0],
            "n_proteins": matrix.shape[1],
            "pca_dimension": pca_dim,
            "pca_variance": pca_var,
            "knn_neighbors": n_neighbors,
            "estimated_K": k_est,
        })

        print(f"{name:<18} K = {k_est}")

    summary = pd.DataFrame(rows).sort_values("dataset")
    summary.to_csv(args.output_dir / "cluster_number_summary.csv", index=False)

    print(f"\nResults saved to {args.output_dir / 'cluster_number_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
