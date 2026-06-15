# -*- coding: utf-8 -*-
"""
Quantitative PCA for metadata-enriched CA1 meta-analysis.

Input:
    ca1_metadata_enriched_features.csv

Outputs:
    ca1_quantitative_pca_coordinates.csv
    ca1_quantitative_pca_loadings.csv
    ca1_quantitative_pca_explained_variance.csv
    ca1_pca_quality_report.csv
    05_quantitative_pca.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


QUANTITATIVE_FEATURES = [
    "best_unit_count",
    "best_spike_count",
    "n_lfp_channels",
    "n_electrodes",
    "n_trials",
    "n_event_times_total",
    "n_position_samples",
    "recording_duration_s",
    "spikes_per_unit",
    "units_per_minute",
    "spikes_per_minute",
    "trials_per_minute",
    "events_per_trial",
    "lfp_channels_per_unit",
    "electrodes_per_unit",
    "cross_dataset_reuse_score",
    "data_analysis_potential_score",
]


LOG_TRANSFORM_FEATURES = [
    "best_unit_count",
    "best_spike_count",
    "n_lfp_channels",
    "n_electrodes",
    "n_trials",
    "n_event_times_total",
    "n_position_samples",
    "recording_duration_s",
    "spikes_per_unit",
    "units_per_minute",
    "spikes_per_minute",
    "trials_per_minute",
    "events_per_trial",
    "lfp_channels_per_unit",
    "electrodes_per_unit",
]


def safe_numeric(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def build_pca_matrix(df):
    available_features = []

    X = pd.DataFrame(index=df.index)

    for feature in QUANTITATIVE_FEATURES:
        if feature in df.columns:
            values = safe_numeric(df, feature)

            if feature in LOG_TRANSFORM_FEATURES:
                values = np.log1p(values.clip(lower=0))

            X[feature] = values
            available_features.append(feature)

    # remove fully missing columns
    X = X.dropna(axis=1, how="all")

    # remove constant columns
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]

    return X


def run_quantitative_pca(input_csv, output_dir=None, n_clusters=4):
    input_csv = Path(input_csv)

    if output_dir is None:
        output_dir = input_csv.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    X = build_pca_matrix(df)

    if X.shape[1] < 2:
        raise ValueError("Not enough quantitative variables for PCA.")

    quality_rows = []

    for col in X.columns:
        quality_rows.append(
            {
                "feature": col,
                "missing_values": int(X[col].isna().sum()),
                "missing_fraction": float(X[col].isna().mean()),
                "n_unique": int(X[col].nunique(dropna=True)),
                "mean": float(X[col].mean(skipna=True)),
                "std": float(X[col].std(skipna=True)),
                "min": float(X[col].min(skipna=True)),
                "max": float(X[col].max(skipna=True)),
                "used_for_pca": True,
            }
        )

    quality_report = pd.DataFrame(quality_rows)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
        }
    )

    metadata_cols = [
        "dataset_short_name",
        "session_id",
        "subject_id",
        "behavioral_context",
        "recording_system",
        "brain_regions",
        "recommended_analysis_type",
        "recording_profile_group",
    ]

    for col in metadata_cols:
        if col in df.columns:
            pca_df[col] = df[col]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    pca_df["quantitative_cluster"] = kmeans.fit_predict(X_scaled)

    if len(set(pca_df["quantitative_cluster"])) > 1:
        sil = silhouette_score(X_scaled, pca_df["quantitative_cluster"])
    else:
        sil = np.nan

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_loading", "PC2_loading"],
        index=X.columns,
    ).reset_index(names="feature")

    loadings["abs_PC1_loading"] = loadings["PC1_loading"].abs()
    loadings["abs_PC2_loading"] = loadings["PC2_loading"].abs()

    explained = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )

    pca_df.to_csv(output_dir / "ca1_quantitative_pca_coordinates.csv", index=False)
    loadings.to_csv(output_dir / "ca1_quantitative_pca_loadings.csv", index=False)
    explained.to_csv(output_dir / "ca1_quantitative_pca_explained_variance.csv", index=False)
    quality_report.to_csv(output_dir / "ca1_pca_quality_report.csv", index=False)

    plt.figure(figsize=(8, 6))

    for dataset_name, group in pca_df.groupby("dataset_short_name"):
        plt.scatter(group["PC1"], group["PC2"], label=dataset_name, alpha=0.75)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.title("Quantitative PCA of metadata-enriched CA1 recording profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "05_quantitative_pca.png", dpi=300)
    plt.close()

    print("Done.")
    print("Input:", input_csv)
    print("Output directory:", output_dir)
    print("Number of quantitative features used:", X.shape[1])
    print("Features used:", list(X.columns))
    print("Explained variance PC1:", pca.explained_variance_ratio_[0])
    print("Explained variance PC2:", pca.explained_variance_ratio_[1])
    print("Silhouette score:", sil)

    print("\nTop PC1 drivers:")
    print(loadings.sort_values("abs_PC1_loading", ascending=False).head(10))

    print("\nTop PC2 drivers:")
    print(loadings.sort_values("abs_PC2_loading", ascending=False).head(10))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_csv",
        help="Path to ca1_metadata_enriched_features.csv",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=4,
        help="Number of KMeans clusters",
    )

    args = parser.parse_args()

    run_quantitative_pca(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
    )


if __name__ == "__main__":
    main()