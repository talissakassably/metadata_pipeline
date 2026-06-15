# -*- coding: utf-8 -*-
"""Clean, presentation-ready plots for the CA1 meta-analysis."""

import numpy as np
import matplotlib.pyplot as plt

from .features import COMPLETENESS_FLAGS


def plot_completeness_by_dataset(dataset_summary, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = dataset_summary.sort_values("mean_metadata_completeness_score", ascending=False)
    ax.bar(plot_df["dataset_short_name"], plot_df["mean_metadata_completeness_score"])
    ax.set_ylabel("Mean metadata completeness score")
    ax.set_title("Metadata completeness by dataset")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "01_completeness_by_dataset.png", dpi=250)
    plt.close(fig)


def plot_missingness_heatmap(df, figures_dir):
    availability = df.groupby("dataset_short_name")[COMPLETENESS_FLAGS].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(availability.values, aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Fraction present")
    ax.set_yticks(np.arange(len(availability.index)), availability.index)
    ax.set_xticks(np.arange(len(availability.columns)), availability.columns, rotation=45, ha="right")
    ax.set_title("Metadata availability heatmap")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_metadata_availability_heatmap.png", dpi=250)
    plt.close(fig)


def plot_quantitative_pca(df, figures_dir):
    if "pca_1" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for dataset in sorted(df["dataset_short_name"].unique()):
        subset = df[df["dataset_short_name"] == dataset]
        ax.scatter(subset["pca_1"], subset["pca_2"], label=dataset, alpha=0.75, s=35)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Quantitative PCA of metadata-enriched CA1 profiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "03_quantitative_pca_profiles.png", dpi=250)
    plt.close(fig)


def plot_pca_loadings(pca_loadings, figures_dir):
    if pca_loadings is None or pca_loadings.empty:
        return
    top = pca_loadings.sort_values("max_abs_loading", ascending=False).head(12).copy()
    top = top.sort_values("max_abs_loading", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["max_abs_loading"])
    ax.set_xlabel("Maximum absolute loading on PC1 or PC2")
    ax.set_title("Top quantitative drivers of PCA dimensions")
    fig.tight_layout()
    fig.savefig(figures_dir / "04_quantitative_pca_top_drivers.png", dpi=250)
    plt.close(fig)


def plot_reuse_score_by_dataset(dataset_summary, figures_dir):
    if "mean_data_analysis_potential_score" not in dataset_summary.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = dataset_summary.sort_values("mean_data_analysis_potential_score", ascending=False)
    ax.bar(plot_df["dataset_short_name"], plot_df["mean_data_analysis_potential_score"])
    ax.set_ylabel("Mean data analysis potential score")
    ax.set_title("Metadata-enriched analysis potential by dataset")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "05_reuse_score_by_dataset.png", dpi=250)
    plt.close(fig)


def plot_acm(mca_coordinates, figures_dir):
    if mca_coordinates is None or mca_coordinates.empty:
        return
    x_col = "mca_1_plot" if "mca_1_plot" in mca_coordinates.columns else "mca_1"
    y_col = "mca_2_plot" if "mca_2_plot" in mca_coordinates.columns else "mca_2"
    fig, ax = plt.subplots(figsize=(9, 7))
    if "dataset_short_name" in mca_coordinates.columns:
        for dataset in sorted(mca_coordinates["dataset_short_name"].unique()):
            subset = mca_coordinates[mca_coordinates["dataset_short_name"] == dataset]
            ax.scatter(subset[x_col], subset[y_col], label=dataset, alpha=0.55, s=35)
        centroids = mca_coordinates.groupby("dataset_short_name")[["mca_1", "mca_2"]].mean()
        for dataset, row in centroids.iterrows():
            ax.scatter(row["mca_1"], row["mca_2"], marker="X", s=180, edgecolor="black", linewidth=1.0)
            ax.text(row["mca_1"], row["mca_2"], "  " + str(dataset), fontsize=9, va="center")
        ax.legend()
    else:
        ax.scatter(mca_coordinates[x_col], mca_coordinates[y_col], alpha=0.55, s=35)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("ACM-like analysis of categorical metadata")
    fig.tight_layout()
    fig.savefig(figures_dir / "06_categorical_acm_profiles.png", dpi=250)
    plt.close(fig)


def plot_acm_variable_importance(variable_importance, figures_dir):
    if variable_importance is None or variable_importance.empty:
        return
    top = variable_importance.sort_values("max_abs_loading", ascending=False).head(12).copy()
    top = top.sort_values("max_abs_loading", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["base_variable"], top["max_abs_loading"])
    ax.set_xlabel("Maximum absolute loading on Dimension 1 or 2")
    ax.set_title("Top categorical metadata drivers of ACM dimensions")
    fig.tight_layout()
    fig.savefig(figures_dir / "07_categorical_acm_top_variables.png", dpi=250)
    plt.close(fig)
