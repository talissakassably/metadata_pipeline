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
    for dataset in sorted(df["dataset_short_name"].unique()):
        subset = df[df["dataset_short_name"] == dataset]
        plt.scatter(subset["pca_1"], subset["pca_2"], label=dataset, alpha=0.8)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Quantitative PCA of metadata-enriched CA1 profiles")
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
    plt.title("Quantitative cluster composition by dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "04_cluster_composition.png", dpi=200)
    plt.close()


def plot_reuse_score_by_dataset(dataset_summary, figures_dir):
    if "mean_data_analysis_potential_score" not in dataset_summary.columns:
        return
    plot_df = dataset_summary.sort_values("mean_data_analysis_potential_score", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["dataset_short_name"], plot_df["mean_data_analysis_potential_score"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean data analysis potential score")
    plt.title("Metadata-enriched analysis potential by dataset")
    plt.tight_layout()
    plt.savefig(figures_dir / "05_reuse_score_by_dataset.png", dpi=200)
    plt.close()


def plot_categorical_mca(mca_coordinates, figures_dir):
    """Improved AFC/ACM-like scatter plot with jitter and centroids."""
    if mca_coordinates is None or mca_coordinates.empty:
        return

    x_col = "mca_1_plot" if "mca_1_plot" in mca_coordinates.columns else "mca_1"
    y_col = "mca_2_plot" if "mca_2_plot" in mca_coordinates.columns else "mca_2"

    plt.figure(figsize=(9, 7))
    if "dataset_short_name" in mca_coordinates.columns:
        for dataset in sorted(mca_coordinates["dataset_short_name"].unique()):
            subset = mca_coordinates[mca_coordinates["dataset_short_name"] == dataset]
            plt.scatter(subset[x_col], subset[y_col], label=dataset, alpha=0.55, s=35)

        centroids = mca_coordinates.groupby("dataset_short_name")[["mca_1", "mca_2"]].mean()
        for dataset, row in centroids.iterrows():
            plt.scatter(row["mca_1"], row["mca_2"], marker="X", s=180, edgecolor="black", linewidth=1.0)
            plt.text(row["mca_1"], row["mca_2"], "  " + str(dataset), fontsize=9, va="center")
        plt.legend()
    else:
        plt.scatter(mca_coordinates[x_col], mca_coordinates[y_col], alpha=0.55, s=35)

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("AFC/ACM-like analysis of categorical metadata, excluding format variables")
    plt.tight_layout()
    plt.savefig(figures_dir / "06_categorical_afc_mca.png", dpi=250)
    plt.close()


def plot_categorical_mca_top_drivers(mca_loadings, figures_dir):
    """Barplot of the strongest category contributors to AFC/ACM dimensions."""
    if mca_loadings is None or mca_loadings.empty:
        return
    if "max_abs_loading" not in mca_loadings.columns:
        return

    top = mca_loadings.sort_values("max_abs_loading", ascending=False).head(15).copy()
    top = top.sort_values("max_abs_loading", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top["category_feature"], top["max_abs_loading"])
    plt.xlabel("Maximum absolute loading on Dimension 1 or 2")
    plt.title("Top categorical metadata drivers of AFC/ACM dimensions")
    plt.tight_layout()
    plt.savefig(figures_dir / "07_categorical_afc_mca_top_drivers.png", dpi=250)
    plt.close()
