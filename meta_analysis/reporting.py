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
            "total_units": group["n_units"].sum(),
            "total_spikes": group["n_spikes_total"].sum(),
            "total_raw_spike_events": group["raw_spike_events_total"].sum(),
            "total_lfp_channels": group["n_lfp_channels"].sum(),
            "total_trials": group["n_trials"].sum(),
            "total_event_times": group["n_event_times_total"].sum(),
            "recording_systems": "; ".join(sorted(set(x for x in group["recording_system"] if x))),
            "metadata_profiles": "; ".join(sorted(set(group["metadata_profile_label"]))),
        })

    return pd.DataFrame(rows)


def make_availability_by_dataset(df):
    rows = []

    for dataset_name, group in df.groupby("dataset_short_name"):
        row = {"dataset_short_name": dataset_name, "n_rows": len(group)}

        for flag in COMPLETENESS_FLAGS + ["extraction_success", "has_standardized_format"]:
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
            "mean_metadata_completeness_score": group["metadata_completeness_score"].mean(),
            "mean_ephys_richness_score": group["ephys_richness_score"].mean(),
            "mean_behavior_richness_score": group["behavior_richness_score"].mean(),
            "mean_openminds_readiness_score": group["openminds_readiness_score"].mean(),
            "mean_n_units": group["n_units"].mean(),
            "mean_n_lfp_channels": group["n_lfp_channels"].mean(),
            "mean_n_trials": group["n_trials"].mean(),
            "mean_n_event_times_total": group["n_event_times_total"].mean(),
        })

    return pd.DataFrame(rows)

def write_interpretation_markdown(output_dir, df, dataset_summary, missing_report, silhouette):
    path = output_dir / "ca1_metadata_ml_case_study_report.md"

    lines = []
    lines.append("# CA1 metadata ML case study report\n")
    lines.append("## Research question\n")
    lines.append("Can automatically extracted metadata be used to compare and cluster heterogeneous CA1-related electrophysiology recording sessions?\n")
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
            f"mean ephys richness={row['mean_ephys_richness_score']:.2f}"
        )
    lines.append("")
    lines.append("## ML outputs\n")
    if "cluster" in df.columns:
        lines.append("- PCA coordinates were computed and saved.")
        lines.append("- KMeans clusters were computed and saved.")
        if silhouette is not None:
            lines.append(f"- Silhouette score: {silhouette:.3f}")
    else:
        lines.append("- PCA/clustering were skipped because scikit-learn was unavailable.")
    lines.append("")
    lines.append("## Important limitation\n")
    lines.append(
        "TouchAndSee may be represented mainly by file/session metadata if the Neo/pickle loading "
        "failed in the provided extraction JSON. Once the legacy pickle compatibility issue is fixed "
        "and the extraction is rerun, the same pipeline will include its internal electrophysiology "
        "metadata automatically if present in the JSON."
    )
    lines.append("")
    lines.append("## Interpretation angle\n")
    lines.append(
        "This analysis treats metadata as a scientific integration layer: because the datasets "
        "are CA1-related but heterogeneous in format and acquisition system, session-level "
        "metadata features can reveal different experimental/recording profiles such as "
        "open-field navigation, task/event-rich recordings, standardized NWB unit-rich sessions, "
        "and metadata-limited sessions."
    )

    path.write_text("\n".join(lines), encoding="utf-8")

