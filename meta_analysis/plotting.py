# -*- coding: utf-8 -*-
"""Plotting functions for CA1 metadata meta-analysis."""

import numpy as np
import matplotlib.pyplot as plt

from .features import COMPLETENESS_FLAGS

def plot_completeness_by_dataset(dataset_summary, figures_dir):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(dataset_summary))
    y = dataset_summary["mean_metadata_completeness_score"].values
    labels = dataset_summary["dataset_short_name"].values

    plt.bar(x, y)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean metadata completeness score")
    plt.title("Metadata completeness by dataset")
    plt.tight_layout()
    plt.savefig(figures_dir / "01_completeness_by_dataset.png", dpi=200)
    plt.close()


def plot_missingness_heatmap(df, figures_dir):
    availability = df.groupby("dataset_short_name")[COMPLETENESS_FLAGS].mean()

    plt.figure(figsize=(12, 5))
    plt.imshow(availability.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Fraction present")
    plt.yticks(np.arange(len(availability.index)), availability.index)
    plt.xticks(np.arange(len(availability.columns)), availability.columns, rotation=45, ha="right")
    plt.title("Metadata availability heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_metadata_availability_heatmap.png", dpi=200)
    plt.close()


def plot_pca(df, figures_dir):
    if "pca_1" not in df.columns:
        return

    plt.figure(figsize=(8, 6))

    datasets = sorted(df["dataset_short_name"].unique())
    for dataset in datasets:
        subset = df[df["dataset_short_name"] == dataset]
        plt.scatter(subset["pca_1"], subset["pca_2"], label=dataset, alpha=0.8)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA of CA1 metadata profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "03_pca_metadata_profiles.png", dpi=200)
    plt.close()


def plot_clusters(df, figures_dir):
    if "cluster" not in df.columns:
        return

    counts = df.groupby(["cluster", "dataset_short_name"]).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(counts))

    x = np.arange(len(counts.index))
    for col in counts.columns:
        values = counts[col].values
        plt.bar(x, values, bottom=bottom, label=col)
        bottom += values

    plt.xticks(x, [str(c) for c in counts.index])
    plt.xlabel("Cluster")
    plt.ylabel("Number of sessions/files")
    plt.title("Cluster composition by dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "04_cluster_composition.png", dpi=200)
    plt.close()

