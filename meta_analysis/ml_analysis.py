# -*- coding: utf-8 -*-
"""Unsupervised ML analysis: PCA and KMeans clustering."""

import numpy as np

def build_ml_matrix(df):
    """
    Build numeric feature matrix for PCA/clustering.
    """

    feature_columns = [
        # Scores
        "metadata_completeness_score",
        "ephys_richness_score",
        "behavior_richness_score",
        "openminds_readiness_score",

        # Log-scaled quantities
        "log_n_units",
        "log_n_spikes_total",
        "log_raw_spike_events_total",
        "log_n_lfp_channels",
        "log_n_electrodes",
        "log_n_trials",
        "log_n_event_times_total",

        # Flags
        "has_subject_metadata",
        "has_session_metadata",
        "has_session_date",
        "has_recording_duration",
        "has_spike_metadata",
        "has_unit_metadata",
        "has_sorted_unit_metadata",
        "has_lfp_metadata",
        "has_trial_metadata",
        "has_event_metadata",
        "has_position_metadata",
        "has_sampling_rate_metadata",
        "has_brain_region_metadata",
        "has_standardized_format",
    ]

    X = df[feature_columns].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)

    return X, feature_columns


def standardize_matrix(X):
    values = X.values.astype(float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    scaled = (values - mean) / std
    return scaled, mean, std


def run_pca_and_clustering(df, n_clusters=4):
    """
    Run PCA and KMeans clustering.

    Requires scikit-learn. If unavailable, the script still produces the
    harmonized CSVs and reports, but skips ML outputs.
    """

    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as error:
        print("scikit-learn is not available. Skipping PCA/clustering.")
        print("Install with: python -m pip install scikit-learn")
        print("Error:", repr(error))
        return df, None, None, None

    X, feature_columns = build_ml_matrix(df)
    X_scaled, _, _ = standardize_matrix(X)

    n_samples = X_scaled.shape[0]
    n_components = 2 if n_samples >= 2 else 1

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    df_ml = df.copy()
    df_ml["pca_1"] = coords[:, 0]
    df_ml["pca_2"] = coords[:, 1] if n_components > 1 else 0.0

    # Ensure sensible cluster number.
    n_clusters = int(min(n_clusters, max(2, n_samples - 1)))
    if n_samples < 3:
        df_ml["cluster"] = 0
        return df_ml, pca, feature_columns, None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    df_ml["cluster"] = labels

    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = None

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("KMeans n_clusters:", n_clusters)
    print("Silhouette score:", sil)

    return df_ml, pca, feature_columns, sil
