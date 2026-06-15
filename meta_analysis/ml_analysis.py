# -*- coding: utf-8 -*-
"""Quantitative PCA/KMeans and categorical ACM-like analysis."""

import re
import numpy as np
import pandas as pd

from .features import QUANTITATIVE_ML_FEATURES, CATEGORICAL_FEATURES


def standardize_matrix(X):
    values = X.values.astype(float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    scaled = (values - mean) / std
    return scaled, mean, std


def build_quantitative_matrix(df):
    used = [c for c in QUANTITATIVE_ML_FEATURES if c in df.columns]
    if len(used) < 2:
        raise ValueError("Not enough quantitative features for PCA.")
    X = df[used].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]
    return X, list(X.columns)


def make_pca_quality_report(X):
    rows = []
    for col in X.columns:
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


def run_quantitative_pca_clustering(df, n_clusters=4):
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as error:
        print("scikit-learn is unavailable. Quantitative PCA skipped.")
        print("Error:", repr(error))
        return df, None, None, None, None, None, None

    X, feature_columns = build_quantitative_matrix(df)
    quality = make_pca_quality_report(X)
    X_scaled, _, _ = standardize_matrix(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    out = df.copy()
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_loading", "PC2_loading"],
        index=feature_columns,
    ).reset_index(names="feature")
    loadings["abs_PC1_loading"] = loadings["PC1_loading"].abs()
    loadings["abs_PC2_loading"] = loadings["PC2_loading"].abs()
    loadings["max_abs_loading"] = loadings[["abs_PC1_loading", "abs_PC2_loading"]].max(axis=1)

    explained = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })

    n_samples = X_scaled.shape[0]
    n_clusters = int(min(n_clusters, max(2, n_samples - 1)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    out["cluster"] = labels

    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = None

    print("Quantitative PCA features used:", feature_columns)
    print("Quantitative PCA explained variance:", pca.explained_variance_ratio_)
    print("Quantitative KMeans silhouette score:", sil)

    return out, pca, feature_columns, sil, loadings, explained, quality


def _clean_cat(value):
    if pd.isna(value):
        return "missing"
    text = str(value).strip()
    return text if text else "missing"


def _category_base(category_feature):
    """Collapse one-hot names like has_lfp_metadata_True into has_lfp_metadata."""
    for suffix in ["_True", "_False", "_missing"]:
        if category_feature.endswith(suffix):
            return category_feature[: -len(suffix)]
    # For labels with many values, remove final value after last underscore only if it is a known generated prefix.
    prefixes = ["metadata_profile_label_", "recommended_analysis_type_", "recording_profile_group_"]
    for prefix in prefixes:
        if category_feature.startswith(prefix):
            return prefix[:-1]
    return re.sub(r"_[^_]+$", "", category_feature)


def _add_jitter(coord_df, X_cat, scale=0.08, random_state=42):
    rng = np.random.default_rng(random_state)
    key = X_cat.astype(str).agg("||".join, axis=1)
    counts = key.map(key.value_counts())
    out = coord_df.copy()
    out["duplicate_profile_count"] = counts.values
    jx = np.zeros(len(out))
    jy = np.zeros(len(out))
    for profile in key[counts > 1].unique():
        idx = np.where(key.values == profile)[0]
        jx[idx] = rng.normal(0, scale, size=len(idx))
        jy[idx] = rng.normal(0, scale, size=len(idx))
    out["mca_1_plot"] = out["mca_1"] + jx
    out["mca_2_plot"] = out["mca_2"] + jy
    return out


def run_categorical_acm(df, n_clusters=4):
    """ACM-like analysis on categorical metadata.

    Strictly, AFC is for two categorical variables and ACM is for several
    categorical variables. Since we use several metadata categories, this is
    ACM-like. We exclude format/source variables by default to avoid trivial
    dataset separation.
    """
    try:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as error:
        print("scikit-learn is unavailable. ACM skipped.")
        print("Error:", repr(error))
        return None, None, None, None, None, None

    used = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if len(used) < 2:
        return None, None, None, None, None, None

    X_cat = df[used].copy()
    quality_rows = []
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].apply(_clean_cat).astype(str)
        vals = sorted(X_cat[col].unique())
        quality_rows.append({
            "feature": col,
            "n_categories": int(X_cat[col].nunique(dropna=False)),
            "categories_preview": "; ".join(vals[:30]),
            "used_for_categorical_analysis": True,
        })
    quality = pd.DataFrame(quality_rows)

    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")

    Z = enc.fit_transform(X_cat)
    names = enc.get_feature_names_out(used)
    Z = pd.DataFrame(Z, columns=names, index=df.index)
    Z = Z.loc[:, Z.nunique(dropna=True) > 1]
    Z = Z.loc[:, Z.sum(axis=0) >= 2]
    if Z.shape[1] < 2:
        return None, None, None, quality, None, None

    Z_scaled, _, _ = standardize_matrix(Z)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Z_scaled)

    coord = pd.DataFrame({"mca_1": coords[:, 0], "mca_2": coords[:, 1]}, index=df.index)
    for col in ["dataset_short_name", "session_id", "subject_id", "recommended_analysis_type", "recording_profile_group"]:
        if col in df.columns:
            coord[col] = df[col].values
    coord = _add_jitter(coord, X_cat)

    n_samples = Z_scaled.shape[0]
    n_clusters = int(min(n_clusters, max(2, n_samples - 1)))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(Z_scaled)
    coord["categorical_cluster"] = labels
    try:
        sil = silhouette_score(Z_scaled, labels)
    except Exception:
        sil = None

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["Dim1_loading", "Dim2_loading"],
        index=Z.columns,
    ).reset_index(names="category_feature")
    loadings["abs_Dim1_loading"] = loadings["Dim1_loading"].abs()
    loadings["abs_Dim2_loading"] = loadings["Dim2_loading"].abs()
    loadings["max_abs_loading"] = loadings[["abs_Dim1_loading", "abs_Dim2_loading"]].max(axis=1)
    loadings["base_variable"] = loadings["category_feature"].apply(_category_base)

    variable_importance = (
        loadings.groupby("base_variable", as_index=False)
        .agg(
            max_abs_loading=("max_abs_loading", "max"),
            top_category=("category_feature", lambda s: s.iloc[0]),
        )
        .sort_values("max_abs_loading", ascending=False)
    )

    explained = pd.DataFrame({
        "dimension": ["Dim1", "Dim2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })

    print("ACM categorical features used:", used)
    print("ACM explained variance:", pca.explained_variance_ratio_)
    print("ACM silhouette score:", sil)

    return coord, loadings, variable_importance, explained, quality, sil
