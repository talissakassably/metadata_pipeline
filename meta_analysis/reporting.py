# -*- coding: utf-8 -*-
"""CSV and markdown reporting functions."""

import numpy as np
import pandas as pd

from .features import COMPLETENESS_FLAGS


def make_dataset_summary(df):
    grouped = df.groupby("dataset_short_name")
    rows = []
    for dataset_name, group in grouped:
        rows.append({
            "dataset_short_name": dataset_name,
            "n_rows": len(group),
            "n_subjects": group["subject_id"].replace("", np.nan).nunique(),
            "n_successful_extractions": int(group["extraction_success"].sum()),
            "mean_metadata_completeness_score": group["metadata_completeness_score"].mean(),
            "mean_ephys_richness_score": group["ephys_richness_score"].mean(),
            "mean_behavior_richness_score": group["behavior_richness_score"].mean(),
            "mean_openminds_readiness_score": group["openminds_readiness_score"].mean(),
            "mean_cross_dataset_reuse_score": group["cross_dataset_reuse_score"].mean(),
            "mean_data_analysis_potential_score": group["data_analysis_potential_score"].mean(),
            "total_units": group["best_unit_count"].sum(),
            "total_spikes": group["best_spike_count"].sum(),
            "total_lfp_channels": group["n_lfp_channels"].sum(),
            "total_trials": group["n_trials"].sum(),
            "total_event_times": group["n_event_times_total"].sum(),
            "n_can_do_spike_analysis": int(group["can_do_spike_analysis"].sum()),
            "n_can_do_lfp_analysis": int(group["can_do_lfp_analysis"].sum()),
            "n_can_do_behavior_analysis": int(group["can_do_behavior_analysis"].sum()),
            "n_can_do_spike_behavior_analysis": int(group["can_do_spike_behavior_analysis"].sum()),
            "n_can_do_cross_dataset_comparison": int(group["can_do_cross_dataset_comparison"].sum()),
            "recording_systems": "; ".join(sorted(set(x for x in group["recording_system"] if x))),
            "metadata_profiles": "; ".join(sorted(set(group["metadata_profile_label"]))),
            "recommended_analysis_types": "; ".join(sorted(set(group["recommended_analysis_type"]))),
        })
    return pd.DataFrame(rows)


def make_availability_by_dataset(df):
    rows = []
    extra_flags = [
        "extraction_success", "has_standardized_format", "can_do_spike_analysis",
        "can_do_lfp_analysis", "can_do_behavior_analysis", "can_do_spike_behavior_analysis",
        "can_do_cross_dataset_comparison",
    ]
    for dataset_name, group in df.groupby("dataset_short_name"):
        row = {"dataset_short_name": dataset_name, "n_rows": len(group)}
        for flag in COMPLETENESS_FLAGS + extra_flags:
            if flag in group.columns:
                row[flag + "_fraction"] = group[flag].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def make_missing_metadata_report(df):
    rows = []
    for dataset_name, group in df.groupby("dataset_short_name"):
        for flag in COMPLETENESS_FLAGS:
            missing_count = int((~group[flag]).sum())
            rows.append({
                "dataset_short_name": dataset_name,
                "field_flag": flag,
                "n_missing": missing_count,
                "n_total": len(group),
                "missing_fraction": missing_count / len(group) if len(group) else 0,
            })
    return pd.DataFrame(rows).sort_values(["dataset_short_name", "missing_fraction"], ascending=[True, False])


def make_cluster_profiles(df):
    if "cluster" not in df.columns:
        return pd.DataFrame()
    rows = []
    for cluster, group in df.groupby("cluster"):
        rows.append({
            "cluster": cluster,
            "n_rows": len(group),
            "datasets": "; ".join(sorted(set(group["dataset_short_name"]))),
            "dominant_profile_labels": "; ".join(sorted(set(group["metadata_profile_label"]))),
            "recommended_analysis_types": "; ".join(sorted(set(group["recommended_analysis_type"]))),
            "mean_metadata_completeness_score": group["metadata_completeness_score"].mean(),
            "mean_ephys_richness_score": group["ephys_richness_score"].mean(),
            "mean_behavior_richness_score": group["behavior_richness_score"].mean(),
            "mean_openminds_readiness_score": group["openminds_readiness_score"].mean(),
            "mean_data_analysis_potential_score": group["data_analysis_potential_score"].mean(),
            "mean_best_unit_count": group["best_unit_count"].mean(),
            "mean_best_spike_count": group["best_spike_count"].mean(),
            "mean_n_lfp_channels": group["n_lfp_channels"].mean(),
            "mean_n_trials": group["n_trials"].mean(),
            "mean_n_event_times_total": group["n_event_times_total"].mean(),
        })
    return pd.DataFrame(rows)


def write_interpretation_markdown(output_dir, df, dataset_summary, missing_report, silhouette, pca_loadings=None, mca_silhouette=None):
    path = output_dir / "ca1_metadata_ml_case_study_report.md"
    lines = []
    lines.append("# CA1 metadata-enriched meta-analysis report\n")
    lines.append("## Research question\n")
    lines.append("Can automatically extracted metadata, including quantitative recording-level variables, be used to compare heterogeneous CA1-related electrophysiology sessions and evaluate their reuse potential?\n")
    lines.append("## Input overview\n")
    lines.append(f"- Total rows in harmonized table: {len(df)}")
    lines.append(f"- Datasets included: {', '.join(sorted(df['dataset_short_name'].unique()))}")
    lines.append("")
    lines.append("## Dataset-level summary\n")
    for _, row in dataset_summary.iterrows():
        lines.append(
            f"- **{row['dataset_short_name']}**: {int(row['n_rows'])} rows, "
            f"{int(row['n_successful_extractions'])} successful extractions, "
            f"mean completeness={row['mean_metadata_completeness_score']:.2f}, "
            f"mean data-analysis potential={row['mean_data_analysis_potential_score']:.2f}"
        )
    lines.append("")
    lines.append("## Quantitative PCA / ML outputs\n")
    if "cluster" in df.columns:
        lines.append("- PCA was computed using quantitative metadata and recording-level features, including unit counts, spike counts, LFP channel counts, trial/event counts, recording duration and derived rates.")
        lines.append("- PCA coordinates, loadings and a PCA quality report were saved.")
        lines.append("- KMeans clusters were computed on the standardized quantitative feature matrix.")
        if silhouette is not None:
            lines.append(f"- Quantitative PCA/KMeans silhouette score: {silhouette:.3f}")
    else:
        lines.append("- PCA/clustering were skipped because scikit-learn was unavailable.")
    if pca_loadings is not None and not pca_loadings.empty:
        top_pc1 = pca_loadings.sort_values("abs_PC1_loading", ascending=False).head(5)["feature"].tolist()
        lines.append(f"- Main PC1 contributors: {', '.join(top_pc1)}")
        if "abs_PC2_loading" in pca_loadings.columns:
            top_pc2 = pca_loadings.sort_values("abs_PC2_loading", ascending=False).head(5)["feature"].tolist()
            lines.append(f"- Main PC2 contributors: {', '.join(top_pc2)}")
    lines.append("")
    lines.append("## AFC/ACM-like categorical analysis\n")
    lines.append("- Categorical metadata were analysed separately with an ACM-like approach based on one-hot encoded metadata categories.")
    if mca_silhouette is not None:
        lines.append(f"- Categorical clustering silhouette score: {mca_silhouette:.3f}")
    lines.append("")
    lines.append("## Interpretation angle\n")
    lines.append("The analysis uses metadata as an integration layer. The updated version no longer relies only on binary metadata presence/absence, but explicitly includes quantitative metadata and recording-level descriptors to evaluate cross-dataset comparability and reuse potential.")
    path.write_text("\n".join(lines), encoding="utf-8")
