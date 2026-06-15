# -*- coding: utf-8 -*-
"""
Biological cross-dataset case study from harmonized biological tables.

This script consumes:
- outputs/biological_tables/harmonized_biological_sessions.csv
- outputs/biological_tables/harmonized_biological_units.csv
- outputs/biological_tables/harmonized_biological_trials.csv

It produces statistical comparisons, PCA and figures.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    out = np.empty(n)
    best = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        best = min(best, ranked[i] * n / rank)
        out[order[i]] = best
    return np.minimum(out, 1.0)


def kruskal_tests(df, group_col, features):
    try:
        from scipy.stats import kruskal
    except Exception:
        return pd.DataFrame()

    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        groups, names = [], []
        for name, sub in df.groupby(group_col):
            vals = pd.to_numeric(sub[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) >= 2:
                groups.append(vals.values)
                names.append(str(name))
        if len(groups) < 2:
            continue
        try:
            h, p = kruskal(*groups)
        except Exception:
            continue
        n = sum(len(g) for g in groups)
        k = len(groups)
        eta = (h - k + 1) / (n - k) if n > k else np.nan
        rows.append({
            "group_col": group_col,
            "feature": feature,
            "groups": "; ".join(names),
            "n_observations": n,
            "kruskal_H": h,
            "p_value": p,
            "eta_squared_approx": max(0, eta) if pd.notna(eta) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_bh"] = bh_adjust(out["p_value"].values)
        out = out.sort_values(["p_value_bh", "eta_squared_approx"], ascending=[True, False])
    return out


def run_pca(df, features):
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x = pd.DataFrame(index=df.index)
    for f in features:
        vals = pd.to_numeric(df.get(f, np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        x["log_" + f] = np.log1p(vals.clip(lower=0))
    x = x.loc[:, x.nunique() > 1]

    if x.shape[1] < 2 or len(x) < 4:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    values = x.values.astype(float)
    values = (values - values.mean(axis=0)) / np.where(values.std(axis=0) == 0, 1, values.std(axis=0))

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(values)

    coord_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1]})
    for col in ["dataset_short_name", "session_id", "subject_id", "behavioral_context", "source_level"]:
        if col in df.columns:
            coord_df[col] = df[col].values

    loadings = pd.DataFrame(pca.components_.T, columns=["PC1_loading", "PC2_loading"])
    loadings.insert(0, "feature", x.columns)
    loadings["abs_PC1_loading"] = loadings["PC1_loading"].abs()
    loadings["abs_PC2_loading"] = loadings["PC2_loading"].abs()
    loadings["max_abs_loading"] = loadings[["abs_PC1_loading", "abs_PC2_loading"]].max(axis=1)
    loadings = loadings.sort_values("max_abs_loading", ascending=False)

    explained = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    return coord_df, loadings, explained


def save_boxplot(df, feature, group_col, title, ylabel, path):
    groups, labels = [], []
    for name, sub in df.groupby(group_col):
        vals = pd.to_numeric(sub[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) > 0:
            groups.append(vals.values)
            labels.append(str(name))
    if not groups:
        return
    plt.figure(figsize=(10, 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def plot_pca(coord_df, path):
    if coord_df.empty:
        return
    plt.figure(figsize=(8, 6))
    for name, sub in coord_df.groupby("dataset_short_name"):
        plt.scatter(sub["PC1"], sub["PC2"], label=name, alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biological case study: PCA of session activity/task profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def plot_pca_drivers(loadings, path):
    if loadings.empty:
        return
    top = loadings.head(12).sort_values("max_abs_loading", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["max_abs_loading"])
    plt.xlabel("Maximum absolute loading on PC1 or PC2")
    plt.title("Biological case study: top PCA drivers")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def write_report(path, sessions, dataset_summary, tests_dataset, tests_context, pca_explained, pca_loadings):
    lines = [
        "# Biological cross-dataset case study",
        "",
        "## Question",
        "Across CA1-related electrophysiology datasets, how do experimental context and recording structure affect available neural activity profiles and reuse potential?",
        "",
        "## Datasets",
    ]
    for _, row in dataset_summary.iterrows():
        lines.append(
            f"- {row['dataset_short_name']}: {int(row['n_sessions'])} sessions, "
            f"median units={row['median_n_units']:.2f}, "
            f"median spikes={row['median_n_spikes_total']:.2f}, "
            f"median trials={row['median_n_trials']:.2f}, "
            f"median completeness={row['median_completeness']:.2f}"
        )
    if not tests_dataset.empty:
        lines.extend(["", "## Main dataset-level differences"])
        for _, row in tests_dataset.head(8).iterrows():
            lines.append(f"- {row['feature']}: H={row['kruskal_H']:.2f}, BH-adjusted p={row['p_value_bh']:.3g}, effect≈{row['eta_squared_approx']:.2f}")
    if not tests_context.empty:
        lines.extend(["", "## Main behavioral-context differences"])
        for _, row in tests_context.head(8).iterrows():
            lines.append(f"- {row['feature']}: H={row['kruskal_H']:.2f}, BH-adjusted p={row['p_value_bh']:.3g}, effect≈{row['eta_squared_approx']:.2f}")
    if not pca_explained.empty:
        lines.extend(["", "## PCA"])
        for _, row in pca_explained.iterrows():
            lines.append(f"- {row['component']}: {row['explained_variance_ratio'] * 100:.1f}% variance explained")
        if not pca_loadings.empty:
            lines.append("Top PCA drivers: " + ", ".join(pca_loadings.head(5)["feature"].astype(str)) + ".")
    lines.extend([
        "",
        "## Interpretation",
        "This case study compares biologically meaningful extracted variables: unit yield, spike yield, firing-rate summaries, trial structure, events, LFP availability and recording duration. Metadata is used to harmonize and contextualize the recording descriptors.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    input_dir = Path(args.input_dir)
    out = Path(args.output_dir)
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    sessions = pd.read_csv(input_dir / "harmonized_biological_sessions.csv")
    units_path = input_dir / "harmonized_biological_units.csv"
    trials_path = input_dir / "harmonized_biological_trials.csv"
    units = pd.read_csv(units_path) if units_path.exists() else pd.DataFrame()
    trials = pd.read_csv(trials_path) if trials_path.exists() else pd.DataFrame()

    features = [
        "n_units", "n_spikes_total", "recording_duration_s",
        "mean_firing_rate_hz_filled", "n_trials", "n_events_total",
        "n_lfp_channels", "spikes_per_unit", "spikes_per_trial",
        "events_per_trial", "biological_table_completeness_score",
    ]

    dataset_summary = sessions.groupby("dataset_short_name").agg(
        n_sessions=("session_id", "nunique"),
        n_subjects=("subject_id", "nunique"),
        median_n_units=("n_units", "median"),
        median_n_spikes_total=("n_spikes_total", "median"),
        median_n_trials=("n_trials", "median"),
        median_n_events=("n_events_total", "median"),
        median_n_lfp_channels=("n_lfp_channels", "median"),
        median_completeness=("biological_table_completeness_score", "median"),
    ).reset_index()
    dataset_summary.to_csv(out / "biological_case_dataset_summary.csv", index=False)

    context_summary = sessions.groupby("behavioral_context").agg(
        n_sessions=("session_id", "nunique"),
        n_datasets=("dataset_short_name", "nunique"),
        median_n_units=("n_units", "median"),
        median_n_spikes_total=("n_spikes_total", "median"),
        median_n_trials=("n_trials", "median"),
        median_completeness=("biological_table_completeness_score", "median"),
    ).reset_index()
    context_summary.to_csv(out / "biological_case_context_summary.csv", index=False)

    tests_dataset = kruskal_tests(sessions, "dataset_short_name", features)
    tests_context = kruskal_tests(sessions, "behavioral_context", features)
    tests_dataset.to_csv(out / "biological_case_kruskal_by_dataset.csv", index=False)
    tests_context.to_csv(out / "biological_case_kruskal_by_context.csv", index=False)

    coords, loadings, explained = run_pca(sessions, features)
    coords.to_csv(out / "biological_case_pca_coordinates.csv", index=False)
    loadings.to_csv(out / "biological_case_pca_loadings.csv", index=False)
    explained.to_csv(out / "biological_case_pca_explained_variance.csv", index=False)

    save_boxplot(sessions, "n_units", "dataset_short_name", "Unit yield by dataset", "Number of units / spiketrains", fig / "01_unit_yield_by_dataset.png")
    save_boxplot(sessions, "n_spikes_total", "dataset_short_name", "Spike yield by dataset", "Total spikes / sorted spike events", fig / "02_spike_yield_by_dataset.png")
    save_boxplot(sessions, "n_trials", "dataset_short_name", "Trial structure by dataset", "Number of trials / segments", fig / "03_trials_by_dataset.png")
    save_boxplot(sessions, "mean_firing_rate_hz_filled", "dataset_short_name", "Firing-rate summaries by dataset", "Mean firing-rate summary / estimate Hz", fig / "04_firing_rate_by_dataset.png")
    plot_pca(coords, fig / "05_session_pca.png")
    plot_pca_drivers(loadings, fig / "06_pca_top_drivers.png")

    write_report(out / "biological_case_study_report.md", sessions, dataset_summary, tests_dataset, tests_context, explained, loadings)

    print("Case study complete.")
    print("Output directory:", out)
    print(dataset_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run biological cross-dataset case study from harmonized biological tables.")
    parser.add_argument("--input_dir", default="outputs/biological_tables")
    parser.add_argument("--output_dir", default="outputs/case_studies/biological_cross_dataset")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
