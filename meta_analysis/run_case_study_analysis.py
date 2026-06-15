# -*- coding: utf-8 -*-
"""
Run an actual metadata-guided CA1 cross-dataset case study.

This script uses the enriched metadata table produced by the meta-analysis
pipeline and turns it into a concrete case study:

Case study question
-------------------
Can metadata-enriched quantitative recording features identify which CA1-related
sessions are reusable for downstream analyses, and which recording properties
drive differences between reusable profiles?

What it does
------------
1. Selects reusable sessions using the metadata-derived reuse flags.
2. Compares quantitative recording features across datasets and analysis types.
3. Runs non-parametric Kruskal-Wallis tests.
4. Computes pairwise dataset comparisons where possible.
5. Builds a simple feature-importance model to identify which quantitative
   variables predict analysis suitability.
6. Computes a dataset similarity/distance matrix based on median quantitative
   recording profiles.
7. Saves clean tables, figures and a short markdown summary.

Inputs
------
    ca1_metadata_enriched_features.csv

Outputs
-------
    case_study/ca1_case_study_reusable_sessions.csv
    case_study/ca1_case_study_dataset_feature_summary.csv
    case_study/ca1_case_study_kw_tests_by_dataset.csv
    case_study/ca1_case_study_kw_tests_by_analysis_type.csv
    case_study/ca1_case_study_pairwise_dataset_tests.csv
    case_study/ca1_case_study_feature_importance.csv
    case_study/ca1_case_study_dataset_distance_matrix.csv
    case_study/ca1_case_study_summary.md

    figures/08_case_study_reusable_sessions_by_analysis_type.png
    figures/09_case_study_top_feature_boxplots.png
    figures/10_case_study_dataset_distance_heatmap.png
    figures/11_case_study_feature_importance.png

Run from repository root
------------------------
    py meta_analysis\\run_case_study_analysis.py ^
        meta_analysis\\outputs\\main\\ca1_metadata_enriched_features.csv ^
        --output_dir meta_analysis\\outputs

If your pipeline does not use the main/ subfolder, run:
    py meta_analysis\\run_case_study_analysis.py ^
        meta_analysis\\outputs\\ca1_metadata_enriched_features.csv ^
        --output_dir meta_analysis\\outputs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IDENTITY_COLUMNS = [
    "dataset_short_name",
    "session_id",
    "subject_id",
    "session_date",
    "behavioral_context",
    "recording_system",
    "brain_regions",
    "recommended_analysis_type",
    "recording_profile_group",
]

REUSE_FLAGS = [
    "can_do_spike_analysis",
    "can_do_lfp_analysis",
    "can_do_behavior_analysis",
    "can_do_position_analysis",
    "can_do_spike_behavior_analysis",
    "can_do_spike_position_analysis",
    "can_do_lfp_behavior_analysis",
    "can_do_cross_dataset_comparison",
]

# Quantitative variables used for the case study.
# We use raw and derived values, and log-transform them internally for tests/ML.
QUANTITATIVE_FEATURES = [
    "best_unit_count",
    "best_spike_count",
    "best_behavior_count",
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

DISPLAY_FEATURE_NAMES = {
    "best_unit_count": "Unit count",
    "best_spike_count": "Spike count",
    "best_behavior_count": "Behavior/event count",
    "n_lfp_channels": "LFP channels",
    "n_electrodes": "Electrodes",
    "n_trials": "Trials",
    "n_event_times_total": "Event times",
    "n_position_samples": "Position samples",
    "recording_duration_s": "Recording duration",
    "spikes_per_unit": "Spikes per unit",
    "units_per_minute": "Units per minute",
    "spikes_per_minute": "Spikes per minute",
    "trials_per_minute": "Trials per minute",
    "events_per_trial": "Events per trial",
    "lfp_channels_per_unit": "LFP channels per unit",
    "electrodes_per_unit": "Electrodes per unit",
    "cross_dataset_reuse_score": "Cross-dataset reuse score",
    "data_analysis_potential_score": "Data analysis potential",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def find_default_input():
    candidates = [
        Path("meta_analysis/outputs/main/ca1_metadata_enriched_features.csv"),
        Path("meta_analysis/outputs/ca1_metadata_enriched_features.csv"),
        Path("outputs/main/ca1_metadata_enriched_features.csv"),
        Path("outputs/ca1_metadata_enriched_features.csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def to_bool(series):
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def ensure_numeric(df, features):
    out = pd.DataFrame(index=df.index)

    for feature in features:
        if feature in df.columns:
            out[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0)
        else:
            out[feature] = 0.0

    return out


def log_transform_matrix(X):
    X_log = X.copy()

    for col in X_log.columns:
        # Log transform positive count/rate variables.
        # Scores between 0 and 1 are kept as-is.
        if col.endswith("_score"):
            X_log[col] = pd.to_numeric(X_log[col], errors="coerce").fillna(0)
        else:
            X_log[col] = np.log1p(pd.to_numeric(X_log[col], errors="coerce").fillna(0).clip(lower=0))

    return X_log


def benjamini_hochberg(p_values):
    """Return BH-adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)

    if n == 0:
        return np.array([])

    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)

    cumulative = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        cumulative = min(cumulative, value)
        adjusted[order[i]] = cumulative

    return np.minimum(adjusted, 1.0)


def kruskal_test_by_group(df, features, group_col, min_groups=2):
    """
    Non-parametric Kruskal-Wallis tests for each feature across group_col.
    """

    try:
        from scipy.stats import kruskal
    except Exception:
        print("scipy not available. Skipping Kruskal-Wallis tests.")
        return pd.DataFrame()

    rows = []

    for feature in features:
        if feature not in df.columns:
            continue

        valid = df[[group_col, feature]].copy()
        valid[feature] = pd.to_numeric(valid[feature], errors="coerce")
        valid = valid.dropna()

        groups = []
        group_names = []

        for name, sub in valid.groupby(group_col):
            values = sub[feature].values
            if len(values) >= 2:
                groups.append(values)
                group_names.append(name)

        if len(groups) < min_groups:
            continue

        try:
            stat, p_value = kruskal(*groups)
        except Exception:
            continue

        n = sum(len(g) for g in groups)
        k = len(groups)

        # Approximate effect size for Kruskal-Wallis.
        eta_squared = (stat - k + 1) / (n - k) if n > k else np.nan
        eta_squared = max(0, eta_squared) if not pd.isna(eta_squared) else np.nan

        rows.append({
            "feature": feature,
            "feature_label": DISPLAY_FEATURE_NAMES.get(feature, feature),
            "group_col": group_col,
            "n_groups": k,
            "groups": "; ".join(str(x) for x in group_names),
            "n_observations": n,
            "kruskal_H": stat,
            "p_value": p_value,
            "eta_squared_approx": eta_squared,
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_value_bh"] = benjamini_hochberg(result["p_value"].values)
        result = result.sort_values(["p_value_bh", "eta_squared_approx"], ascending=[True, False])

    return result


def pairwise_mannwhitney(df, features, group_col):
    """
    Pairwise Mann-Whitney U tests between groups for each feature.
    Useful as a supplement after Kruskal-Wallis.
    """

    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        print("scipy not available. Skipping pairwise tests.")
        return pd.DataFrame()

    rows = []
    groups = sorted(df[group_col].dropna().unique())

    for feature in features:
        if feature not in df.columns:
            continue

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1 = groups[i]
                g2 = groups[j]

                x = pd.to_numeric(df.loc[df[group_col] == g1, feature], errors="coerce").dropna().values
                y = pd.to_numeric(df.loc[df[group_col] == g2, feature], errors="coerce").dropna().values

                if len(x) < 2 or len(y) < 2:
                    continue

                try:
                    stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
                except Exception:
                    continue

                rows.append({
                    "feature": feature,
                    "feature_label": DISPLAY_FEATURE_NAMES.get(feature, feature),
                    "group_col": group_col,
                    "group_1": g1,
                    "group_2": g2,
                    "n_1": len(x),
                    "n_2": len(y),
                    "median_1": float(np.median(x)),
                    "median_2": float(np.median(y)),
                    "median_difference_1_minus_2": float(np.median(x) - np.median(y)),
                    "mannwhitney_U": stat,
                    "p_value": p_value,
                })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_value_bh"] = benjamini_hochberg(result["p_value"].values)
        result = result.sort_values(["p_value_bh", "feature", "group_1", "group_2"])

    return result


def summarize_by_dataset(df, features):
    rows = []

    for dataset, group in df.groupby("dataset_short_name"):
        row = {
            "dataset_short_name": dataset,
            "n_sessions": len(group),
        }

        for flag in REUSE_FLAGS:
            if flag in group.columns:
                row[flag + "_n"] = int(group[flag].sum())
                row[flag + "_fraction"] = float(group[flag].mean())

        for feature in features:
            if feature in group.columns:
                values = pd.to_numeric(group[feature], errors="coerce").fillna(0)
                row[feature + "_mean"] = float(values.mean())
                row[feature + "_median"] = float(values.median())
                row[feature + "_std"] = float(values.std())

        rows.append(row)

    return pd.DataFrame(rows)


def compute_dataset_distance_matrix(df, features):
    """
    Dataset distance matrix based on standardized median quantitative profiles.
    """

    X = ensure_numeric(df, features)
    X_log = log_transform_matrix(X)
    X_log["dataset_short_name"] = df["dataset_short_name"].values

    medians = X_log.groupby("dataset_short_name").median()

    # Standardize across datasets/features.
    values = medians.values.astype(float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    scaled = (values - mean) / std

    names = list(medians.index)
    dist = np.zeros((len(names), len(names)))

    for i in range(len(names)):
        for j in range(len(names)):
            dist[i, j] = np.linalg.norm(scaled[i] - scaled[j])

    return pd.DataFrame(dist, index=names, columns=names), medians


def random_forest_feature_importance(df, features, target_col):
    """
    Feature importance for predicting analysis type/reuse profile from quantitative features.
    This is not used as a biological classifier; it explains which quantitative
    variables best predict reuse profiles.
    """

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score, StratifiedKFold
    except Exception:
        print("scikit-learn unavailable. Skipping feature importance.")
        return pd.DataFrame(), np.nan

    if target_col not in df.columns:
        return pd.DataFrame(), np.nan

    valid = df.copy()
    valid[target_col] = valid[target_col].fillna("missing").astype(str)

    # Need at least 2 classes and enough examples per class.
    counts = valid[target_col].value_counts()
    kept_classes = counts[counts >= 3].index
    valid = valid[valid[target_col].isin(kept_classes)].copy()

    if valid[target_col].nunique() < 2 or len(valid) < 10:
        return pd.DataFrame(), np.nan

    X = ensure_numeric(valid, features)
    X_log = log_transform_matrix(X)

    # Remove constant features.
    X_log = X_log.loc[:, X_log.nunique(dropna=True) > 1]

    if X_log.shape[1] < 2:
        return pd.DataFrame(), np.nan

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(valid[target_col].values)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=4,
    )

    cv_score = np.nan

    try:
        min_class = np.bincount(y).min()
        n_splits = int(min(5, min_class))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_score = float(cross_val_score(clf, X_log, y, cv=cv, scoring="balanced_accuracy").mean())
    except Exception:
        cv_score = np.nan

    clf.fit(X_log, y)

    importance = pd.DataFrame({
        "feature": X_log.columns,
        "feature_label": [DISPLAY_FEATURE_NAMES.get(x, x) for x in X_log.columns],
        "importance": clf.feature_importances_,
        "target": target_col,
        "cv_balanced_accuracy": cv_score,
    }).sort_values("importance", ascending=False)

    return importance, cv_score


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_reusable_sessions(df, figures_dir):
    counts = []

    for flag in REUSE_FLAGS:
        if flag in df.columns:
            counts.append({
                "analysis_type": flag.replace("can_do_", "").replace("_", " "),
                "n_sessions": int(df[flag].sum()),
            })

    plot_df = pd.DataFrame(counts).sort_values("n_sessions", ascending=True)

    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["analysis_type"], plot_df["n_sessions"])
    plt.xlabel("Number of reusable sessions")
    plt.title("Case study: reusable CA1 sessions by analysis type")
    plt.tight_layout()
    plt.savefig(figures_dir / "08_case_study_reusable_sessions_by_analysis_type.png", dpi=250)
    plt.close()


def plot_top_feature_boxplots(df, top_features, figures_dir):
    top_features = [f for f in top_features if f in df.columns]

    if len(top_features) == 0:
        return

    n_features = min(4, len(top_features))
    selected = top_features[:n_features]

    datasets = sorted(df["dataset_short_name"].dropna().unique())

    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5), squeeze=False)

    for ax, feature in zip(axes[0], selected):
        values = [
            pd.to_numeric(df.loc[df["dataset_short_name"] == dataset, feature], errors="coerce").fillna(0).values
            for dataset in datasets
        ]

        ax.boxplot(values, labels=datasets, showfliers=False)
        ax.set_title(DISPLAY_FEATURE_NAMES.get(feature, feature))
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_ylabel("Value")

    fig.suptitle("Case study: top quantitative features across datasets")
    plt.tight_layout()
    plt.savefig(figures_dir / "09_case_study_top_feature_boxplots.png", dpi=250)
    plt.close()


def plot_distance_heatmap(distance_matrix, figures_dir):
    if distance_matrix.empty:
        return

    plt.figure(figsize=(7, 6))
    plt.imshow(distance_matrix.values, aspect="auto")
    plt.colorbar(label="Euclidean distance between median standardized profiles")
    plt.xticks(np.arange(len(distance_matrix.columns)), distance_matrix.columns, rotation=30, ha="right")
    plt.yticks(np.arange(len(distance_matrix.index)), distance_matrix.index)
    plt.title("Case study: dataset distance based on quantitative recording profiles")
    plt.tight_layout()
    plt.savefig(figures_dir / "10_case_study_dataset_distance_heatmap.png", dpi=250)
    plt.close()


def plot_feature_importance(importance, figures_dir):
    if importance is None or importance.empty:
        return

    top = importance.sort_values("importance", ascending=False).head(12)
    top = top.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature_label"], top["importance"])
    plt.xlabel("Random forest feature importance")
    plt.title("Case study: variables predicting analysis suitability")
    plt.tight_layout()
    plt.savefig(figures_dir / "11_case_study_feature_importance.png", dpi=250)
    plt.close()


# ---------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------

def write_summary(
    path,
    df,
    reusable_df,
    kw_dataset,
    kw_analysis,
    importance,
    distance_matrix,
    rf_score,
):
    lines = []

    lines.append("# CA1 metadata-guided cross-dataset case study\n")
    lines.append("## Case study question\n")
    lines.append(
        "Can metadata-enriched quantitative recording features identify which CA1-related "
        "sessions are reusable for downstream analyses, and which recording properties "
        "drive differences between reusable profiles?\n"
    )

    lines.append("## Dataset overview\n")
    lines.append(f"- Total sessions/files in enriched table: {len(df)}")
    lines.append(f"- Sessions with at least one reuse possibility: {len(reusable_df)}")
    lines.append(f"- Datasets included: {', '.join(sorted(df['dataset_short_name'].dropna().unique()))}")
    lines.append("")

    lines.append("## Reuse potential\n")
    for flag in REUSE_FLAGS:
        if flag in df.columns:
            label = flag.replace("can_do_", "").replace("_", " ")
            lines.append(f"- {label}: {int(df[flag].sum())} sessions/files")
    lines.append("")

    if not kw_dataset.empty:
        lines.append("## Quantitative features differing across datasets\n")
        top = kw_dataset.head(8)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['feature_label']}: Kruskal-Wallis H={row['kruskal_H']:.2f}, "
                f"BH-adjusted p={row['p_value_bh']:.3g}, "
                f"effect size≈{row['eta_squared_approx']:.2f}"
            )
        lines.append("")

    if not kw_analysis.empty:
        lines.append("## Quantitative features differing across analysis suitability groups\n")
        top = kw_analysis.head(8)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['feature_label']}: Kruskal-Wallis H={row['kruskal_H']:.2f}, "
                f"BH-adjusted p={row['p_value_bh']:.3g}, "
                f"effect size≈{row['eta_squared_approx']:.2f}"
            )
        lines.append("")

    if importance is not None and not importance.empty:
        lines.append("## Variables predicting analysis suitability\n")
        if not pd.isna(rf_score):
            lines.append(f"- Cross-validated balanced accuracy: {rf_score:.2f}")
        top = importance.head(8)
        for _, row in top.iterrows():
            lines.append(f"- {row['feature_label']}: importance={row['importance']:.3f}")
        lines.append("")

    if distance_matrix is not None and not distance_matrix.empty:
        lines.append("## Dataset similarity\n")
        # Most similar non-diagonal pair
        pairs = []
        names = list(distance_matrix.index)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((names[i], names[j], distance_matrix.iloc[i, j]))
        if pairs:
            pairs = sorted(pairs, key=lambda x: x[2])
            closest = pairs[0]
            farthest = pairs[-1]
            lines.append(
                f"- Closest median quantitative profiles: {closest[0]} and {closest[1]} "
                f"(distance={closest[2]:.2f})"
            )
            lines.append(
                f"- Most distant median quantitative profiles: {farthest[0]} and {farthest[1]} "
                f"(distance={farthest[2]:.2f})"
            )
        lines.append("")

    lines.append("## Interpretation\n")
    lines.append(
        "This case study moves beyond descriptive metadata completeness. Metadata is used "
        "to define which sessions are reusable, while quantitative recording-level features "
        "are used to compare datasets and analysis-suitability profiles. The result is a "
        "metadata-guided data reuse analysis rather than a purely administrative metadata summary."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main case study
# ---------------------------------------------------------------------

def run_case_study(input_csv, output_dir=None):
    input_csv = Path(input_csv)

    if output_dir is None:
        output_dir = input_csv.parent
    else:
        output_dir = Path(output_dir)

    case_dir = output_dir / "case_study"
    figures_dir = output_dir / "figures"

    case_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if "dataset_short_name" not in df.columns:
        raise ValueError("Input must contain dataset_short_name column.")

    for flag in REUSE_FLAGS:
        if flag in df.columns:
            df[flag] = to_bool(df[flag])
        else:
            df[flag] = False

    # Standardize text columns
    for col in IDENTITY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    X = ensure_numeric(df, QUANTITATIVE_FEATURES)
    X_log = log_transform_matrix(X)

    # Add log features for statistical testing/modeling.
    analysis_df = df.copy()
    for col in X_log.columns:
        analysis_df["log_" + col] = X_log[col].values

    log_features = ["log_" + f for f in QUANTITATIVE_FEATURES if "log_" + f in analysis_df.columns]

    # Reusable subset = at least one meaningful analysis flag true.
    reuse_any = df[REUSE_FLAGS].any(axis=1)
    reusable_df = analysis_df[reuse_any].copy()

    if reusable_df.empty:
        reusable_df = analysis_df.copy()

    # Summary tables.
    dataset_summary = summarize_by_dataset(df, QUANTITATIVE_FEATURES)
    dataset_summary.to_csv(case_dir / "ca1_case_study_dataset_feature_summary.csv", index=False)

    reusable_cols = [col for col in IDENTITY_COLUMNS + REUSE_FLAGS + QUANTITATIVE_FEATURES if col in df.columns]
    df.loc[reuse_any, reusable_cols].to_csv(case_dir / "ca1_case_study_reusable_sessions.csv", index=False)

    # Statistical tests.
    kw_dataset = kruskal_test_by_group(analysis_df, log_features, "dataset_short_name")
    kw_dataset.to_csv(case_dir / "ca1_case_study_kw_tests_by_dataset.csv", index=False)

    kw_analysis = pd.DataFrame()
    if "recommended_analysis_type" in analysis_df.columns:
        kw_analysis = kruskal_test_by_group(analysis_df, log_features, "recommended_analysis_type")
        kw_analysis.to_csv(case_dir / "ca1_case_study_kw_tests_by_analysis_type.csv", index=False)

    pairwise = pairwise_mannwhitney(analysis_df, log_features, "dataset_short_name")
    pairwise.to_csv(case_dir / "ca1_case_study_pairwise_dataset_tests.csv", index=False)

    # Dataset distance matrix.
    distance_matrix, dataset_medians = compute_dataset_distance_matrix(df, QUANTITATIVE_FEATURES)
    distance_matrix.to_csv(case_dir / "ca1_case_study_dataset_distance_matrix.csv")
    dataset_medians.to_csv(case_dir / "ca1_case_study_dataset_median_profiles.csv")

    # Feature importance for predicting analysis suitability.
    importance, rf_score = random_forest_feature_importance(
        analysis_df,
        log_features,
        target_col="recommended_analysis_type",
    )
    importance.to_csv(case_dir / "ca1_case_study_feature_importance.csv", index=False)

    # Plots.
    plot_reusable_sessions(df, figures_dir)

    if not kw_dataset.empty:
        top_raw_features = [
            row["feature"].replace("log_", "")
            for _, row in kw_dataset.head(4).iterrows()
        ]
    else:
        top_raw_features = ["best_unit_count", "best_spike_count", "n_lfp_channels", "n_trials"]

    plot_top_feature_boxplots(df, top_raw_features, figures_dir)
    plot_distance_heatmap(distance_matrix, figures_dir)
    plot_feature_importance(importance, figures_dir)

    # Summary markdown.
    write_summary(
        case_dir / "ca1_case_study_summary.md",
        df=df,
        reusable_df=reusable_df,
        kw_dataset=kw_dataset,
        kw_analysis=kw_analysis,
        importance=importance,
        distance_matrix=distance_matrix,
        rf_score=rf_score,
    )

    print("Done.")
    print("Input:", input_csv)
    print("Case study outputs:", case_dir)
    print("Figures:", figures_dir)

    print("\nMain case study results:")
    print("Total rows:", len(df))
    print("Reusable rows:", int(reuse_any.sum()))

    if not kw_dataset.empty:
        print("\nTop dataset-level quantitative differences:")
        print(kw_dataset.head(8).to_string(index=False))

    if importance is not None and not importance.empty:
        print("\nTop variables predicting analysis suitability:")
        print(importance.head(8).to_string(index=False))

    return {
        "df": df,
        "dataset_summary": dataset_summary,
        "kw_dataset": kw_dataset,
        "kw_analysis": kw_analysis,
        "importance": importance,
        "distance_matrix": distance_matrix,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a metadata-guided CA1 cross-dataset case study analysis."
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help="Path to ca1_metadata_enriched_features.csv. If omitted, common default paths are tried.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: same folder as input CSV.",
    )

    args = parser.parse_args()

    if args.input_csv is None:
        default_input = find_default_input()
        if default_input is None:
            raise FileNotFoundError(
                "Could not find ca1_metadata_enriched_features.csv. "
                "Please provide it explicitly."
            )
        input_csv = default_input
    else:
        input_csv = Path(args.input_csv)

    run_case_study(
        input_csv=input_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
