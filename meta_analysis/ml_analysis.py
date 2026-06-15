# -*- coding: utf-8 -*-
"""Unsupervised ML analysis: quantitative PCA, KMeans, and AFC/ACM-like analysis."""

import numpy as np
import pandas as pd

from .features import QUANTITATIVE_ML_FEATURES, CATEGORICAL_FEATURES


def build_ml_matrix(df):
    """Build quantitative feature matrix for PCA/clustering."""
    used_features = [col for col in QUANTITATIVE_ML_FEATURES if col in df.columns]
    if len(used_features) < 2:
        raise ValueError("Not enough quantitative features for PCA.")
    X = df[used_features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)
    # Remove constant columns after conversion.
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]
    return X, list(X.columns)


def standardize_matrix(X):
    values = X.values.astype(float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    scaled = (values - mean) / std
    return scaled, mean, std


def make_pca_quality_report(X, feature_columns):
    rows = []
    for col in feature_columns:
        rows.append({
            "feature": col,
            "missing_values": int(X[col].isna().sum()),
            "missing_fraction": float(X[col].isna().mean()),
            "n_unique": int(X[col].nunique(dropna=True)),
            "mean": float(X[col].mean(skipna=True)),
            "std": float(X[col].std(skipna=True)),
            "min": float(X[col].min(skipna=True)),
            "max": float(X[col].max(skipna=True)),
            "used_for_pca": True,
        })
    return pd.DataFrame(rows)


def run_pca_and_clustering(df, n_clusters=4):
    """Run PCA and KMeans on quantitative metadata-enriched variables."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as error:
        print("scikit-learn is not available. Skipping PCA/clustering.")
        print("Install with: python -m pip install scikit-learn")
        print("Error:", repr(error))
        return df, None, None, None, None, None, None

    X, feature_columns = build_ml_matrix(df)
    quality_report = make_pca_quality_report(X, feature_columns)
    X_scaled, _, _ = standardize_matrix(X)

    n_samples = X_scaled.shape[0]
    n_components = 2 if n_samples >= 2 else 1

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    df_ml = df.copy()
    df_ml["pca_1"] = coords[:, 0]
    df_ml["pca_2"] = coords[:, 1] if n_components > 1 else 0.0

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_loading", "PC2_loading"][:n_components],
        index=feature_columns,
    ).reset_index(names="feature")
    if "PC1_loading" in loadings:
        loadings["abs_PC1_loading"] = loadings["PC1_loading"].abs()
    if "PC2_loading" in loadings:
        loadings["abs_PC2_loading"] = loadings["PC2_loading"].abs()

    explained = pd.DataFrame({
        "component": ["PC{}".format(i + 1) for i in range(n_components)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })

    n_clusters = int(min(n_clusters, max(2, n_samples - 1)))
    if n_samples < 3:
        df_ml["cluster"] = 0
        return df_ml, pca, feature_columns, None, loadings, explained, quality_report

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    df_ml["cluster"] = labels

    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = None

    print("Quantitative PCA features used:", feature_columns)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("KMeans n_clusters:", n_clusters)
    print("Silhouette score:", sil)

    return df_ml, pca, feature_columns, sil, loadings, explained, quality_report


def run_categorical_afc_mca(df, n_clusters=4):
    """Run ACM-like analysis using one-hot categorical metadata + PCA.

    Strictly speaking, AFC is for a contingency table and ACM is for multiple
    categorical variables. Because this pipeline uses multiple categorical
    metadata columns, this is an ACM-like analysis.
    """
    try:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as error:
        print("scikit-learn is not available. Skipping AFC/ACM-like analysis.")
        print("Error:", repr(error))
        return None, None, None, None, None

    used_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    if len(used_features) < 2:
        return None, None, None, None, None

    X_cat = df[used_features].copy()
    quality_rows = []
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna("missing").astype(str)
        quality_rows.append({
            "feature": col,
            "n_categories": int(X_cat[col].nunique(dropna=False)),
            "categories_preview": "; ".join(sorted(X_cat[col].unique())[:25]),
            "used_for_categorical_analysis": True,
        })
    quality_report = pd.DataFrame(quality_rows)

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    X_encoded = encoder.fit_transform(X_cat)
    feature_names = encoder.get_feature_names_out(used_features)
    X_df = pd.DataFrame(X_encoded, columns=feature_names, index=df.index)
    nunique = X_df.nunique(dropna=True)
    X_df = X_df.loc[:, nunique > 1]

    X_scaled, _, _ = standardize_matrix(X_df)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    coord_df = pd.DataFrame({"mca_1": coords[:, 0], "mca_2": coords[:, 1]})
    for col in ["dataset_short_name", "session_id", "subject_id", "behavioral_context", "recording_profile_group", "recommended_analysis_type"]:
        if col in df.columns:
            coord_df[col] = df[col]

    n_samples = X_scaled.shape[0]
    n_clusters = int(min(n_clusters, max(2, n_samples - 1)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    coord_df["categorical_cluster"] = labels

    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = None

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["Dim1_loading", "Dim2_loading"],
        index=X_df.columns,
    ).reset_index(names="category_feature")
    loadings["abs_Dim1_loading"] = loadings["Dim1_loading"].abs()
    loadings["abs_Dim2_loading"] = loadings["Dim2_loading"].abs()

    explained = pd.DataFrame({
        "dimension": ["Dim1", "Dim2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })

    print("AFC/ACM categorical features used:", used_features)
    print("AFC/ACM explained variance ratio:", pca.explained_variance_ratio_)
    print("AFC/ACM silhouette score:", sil)

    return coord_df, loadings, explained, quality_report, sil
