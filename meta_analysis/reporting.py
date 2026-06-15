# -*- coding: utf-8 -*-
"""Reporting utilities for the CA1 metadata-enriched meta-analysis."""

import numpy as np
import pandas as pd

from .features import COMPLETENESS_FLAGS


def make_dataset_summary(df):
    rows = []
    for dataset_name, group in df.groupby("dataset_short_name"):
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
            "recommended_analysis_types": "; ".join(sorted(set(group["recommended_analysis_type"]))),
        })
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


def make_reuse_recommendations(df):
    cols = [
        "dataset_short_name", "session_id", "subject_id", "session_date",
        "behavioral_context", "brain_regions", "best_unit_count", "best_spike_count",
        "n_lfp_channels", "n_trials", "n_event_times_total", "recording_duration_s",
        "spikes_per_unit", "spikes_per_minute", "can_do_spike_analysis",
        "can_do_lfp_analysis", "can_do_behavior_analysis", "can_do_position_analysis",
        "can_do_spike_behavior_analysis", "can_do_spike_position_analysis",
        "can_do_lfp_behavior_analysis", "can_do_cross_dataset_comparison",
        "data_analysis_potential_score", "recommended_analysis_type", "missing_requirements",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def write_narrative_summary(output_dir, df, dataset_summary, pca_loadings=None, pca_explained=None, acm_variable_importance=None, quantitative_silhouette=None, acm_silhouette=None):
    path = output_dir / "main" / "ca1_meta_analysis_summary.md"
    lines = []
    lines.append("# CA1 metadata-enriched meta-analysis summary\n")
    lines.append("## Aim\n")
    lines.append("Use automatically extracted metadata as an integration layer to compare heterogeneous CA1-related electrophysiology datasets, quantify reuse potential, and identify the variables driving differences between recording/session profiles.\n")
    lines.append("## Main outputs\n")
    lines.append(f"- Harmonized/enriched sessions: {len(df)}")
    lines.append(f"- Datasets: {', '.join(sorted(df['dataset_short_name'].unique()))}")
    if pca_explained is not None and not pca_explained.empty:
        ratios = pca_explained['explained_variance_ratio'].tolist()
        if len(ratios) >= 2:
            lines.append(f"- Quantitative PCA: PC1={ratios[0]*100:.1f}% and PC2={ratios[1]*100:.1f}% of variance")
    if quantitative_silhouette is not None:
        lines.append(f"- Quantitative clustering silhouette score: {quantitative_silhouette:.3f}")
    if acm_silhouette is not None:
        lines.append(f"- Categorical ACM-like clustering silhouette score: {acm_silhouette:.3f}")
    lines.append("")
    lines.append("## Dataset-level reuse summary\n")
    for _, row in dataset_summary.iterrows():
        lines.append(
            f"- **{row['dataset_short_name']}**: {int(row['n_rows'])} rows, "
            f"mean completeness={row['mean_metadata_completeness_score']:.2f}, "
            f"mean data-analysis potential={row['mean_data_analysis_potential_score']:.2f}, "
            f"spike-behavior reusable sessions={int(row['n_can_do_spike_behavior_analysis'])}."
        )
    lines.append("")
    if pca_loadings is not None and not pca_loadings.empty:
        top_pc1 = pca_loadings.sort_values('abs_PC1_loading', ascending=False).head(5)['feature'].tolist()
        top_pc2 = pca_loadings.sort_values('abs_PC2_loading', ascending=False).head(5)['feature'].tolist()
        lines.append("## Quantitative PCA interpretation\n")
        lines.append("The PCA was computed on standardized quantitative metadata / recording-level descriptors, not only binary metadata flags.")
        lines.append(f"- Main PC1 drivers: {', '.join(top_pc1)}")
        lines.append(f"- Main PC2 drivers: {', '.join(top_pc2)}")
        lines.append("")
    if acm_variable_importance is not None and not acm_variable_importance.empty:
        top = acm_variable_importance.head(6)['base_variable'].tolist()
        lines.append("## Categorical ACM-like interpretation\n")
        lines.append("The categorical analysis was run separately on metadata categories and reuse flags. Format/source variables were excluded from the default analysis to avoid a trivial dataset-format separation.")
        lines.append(f"- Main categorical drivers: {', '.join(top)}")
        lines.append("")
    lines.append("## One-sentence interpretation\n")
    lines.append("The analysis shows that even among biologically related CA1 datasets, metadata completeness, quantitative recording structure, and available behavioral/electrophysiological descriptors strongly determine whether sessions can be reused for cross-dataset analyses.")
    path.write_text("\n".join(lines), encoding="utf-8")
