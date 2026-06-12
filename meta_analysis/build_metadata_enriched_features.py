# -*- coding: utf-8 -*-
"""
Build metadata-enriched features for cross-dataset CA1 electrophysiology analysis.

Input:
    ca1_harmonized_sessions.csv

Output:
    ca1_metadata_enriched_features.csv
    ca1_reuse_recommendations.csv
    ca1_feature_availability_report.csv

Goal:
    Move from simple metadata profiling to metadata-guided data reuse.

Run:
    py meta_analysis\\build_metadata_enriched_features.py ^
        meta_analysis\\outputs\\ca1_harmonized_sessions.csv ^
        --output_dir meta_analysis\\outputs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def to_bool_series(series):
    """
    Convert mixed boolean/string/numeric values to booleans.
    """

    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def safe_numeric(df, column, default=0.0):
    """
    Return a numeric column. If absent, create a default series.
    """

    if column not in df.columns:
        return pd.Series(default, index=df.index)

    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def safe_text(df, column):
    """
    Return a string column. If absent, create an empty string series.
    """

    if column not in df.columns:
        return pd.Series("", index=df.index)

    return df[column].fillna("").astype(str)


def safe_ratio(numerator, denominator):
    """
    Safe division. Works when denominator is either a Series or a scalar.
    Returns 0 if denominator is missing or zero.
    """

    numerator = pd.to_numeric(numerator, errors="coerce")

    if isinstance(denominator, pd.Series):
        denominator = pd.to_numeric(denominator, errors="coerce")
    else:
        denominator = pd.Series(denominator, index=numerator.index)
        denominator = pd.to_numeric(denominator, errors="coerce")

    numerator = numerator.fillna(0)
    denominator = denominator.fillna(0)

    ratio = np.where(denominator > 0, numerator / denominator, 0)

    return pd.Series(ratio, index=numerator.index).replace([np.inf, -np.inf], 0).fillna(0)


def make_missing_requirements(row):
    """
    Human-readable explanation of what blocks stronger reuse.
    """

    missing = []

    if not row["has_subject_metadata"]:
        missing.append("subject metadata")

    if not row["has_session_metadata"]:
        missing.append("session metadata")

    if not row["has_session_date"]:
        missing.append("session date")

    if not row["has_spike_metadata"]:
        missing.append("spike/unit metadata")

    if not row["has_lfp_metadata"]:
        missing.append("LFP metadata")

    if (
        not row["has_trial_metadata"]
        and not row["has_event_metadata"]
        and not row["has_position_metadata"]
    ):
        missing.append("behavioral metadata")

    if not row["has_sampling_rate_metadata"]:
        missing.append("sampling-rate metadata")

    if not row["has_brain_region_metadata"]:
        missing.append("brain-region metadata")

    if len(missing) == 0:
        return "none"

    return "; ".join(missing)


def recommend_analysis_type(row):
    """
    Assign the most suitable analysis type from available data.
    """

    if row["can_do_spike_behavior_analysis"]:
        return "spike-behavior analysis"

    if row["can_do_spike_position_analysis"]:
        return "spike-position / navigation analysis"

    if row["can_do_lfp_behavior_analysis"]:
        return "LFP-behavior analysis"

    if row["can_do_spike_analysis"]:
        return "spike/unit analysis"

    if row["can_do_lfp_analysis"]:
        return "LFP metadata-level analysis"

    if row["can_do_behavior_analysis"]:
        return "behavior-only metadata analysis"

    return "metadata curation only"


# ---------------------------------------------------------------------
# Main feature construction
# ---------------------------------------------------------------------


def build_metadata_enriched_features(input_csv, output_dir=None):
    input_csv = Path(input_csv)

    if output_dir is None:
        output_dir = input_csv.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # -----------------------------------------------------------------
    # Identity / descriptive columns
    # -----------------------------------------------------------------

    identity_columns = [
        "dataset_short_name",
        "dataset_name",
        "source_type",
        "source_format",
        "session_id",
        "subject_id",
        "session_date",
        "recording_system",
        "behavioral_context",
        "brain_regions",
        "unit_source",
        "error",
    ]

    for column in identity_columns:
        df[column] = safe_text(df, column)

    # -----------------------------------------------------------------
    # Boolean metadata flags
    # -----------------------------------------------------------------

    bool_columns = [
        "extraction_success",
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
        "has_openminds_candidate_metadata",
    ]

    for column in bool_columns:
        if column not in df.columns:
            df[column] = False
        df[column] = to_bool_series(df[column])

    # -----------------------------------------------------------------
    # Quantitative electrophysiology/session features
    # -----------------------------------------------------------------

    numeric_columns = [
        "recording_duration_s",
        "n_units",
        "n_spiketrains",
        "n_spikes_total",
        "raw_spike_events_total",
        "sorted_units_mclust",
        "sorted_unit_spikes_total_mclust",
        "n_lfp_channels",
        "n_electrodes",
        "n_trials",
        "n_event_objects",
        "n_event_times_total",
        "n_position_samples",
        "n_files",
        "file_size_mb",
    ]

    for column in numeric_columns:
        df[column] = safe_numeric(df, column, default=0)

    # Best available quantitative features across heterogeneous formats.
    # Some datasets store curated unit spikes, others raw spike events.
    df["best_unit_count"] = df[
        [
            "n_units",
            "n_spiketrains",
            "sorted_units_mclust",
        ]
    ].max(axis=1)

    df["best_spike_count"] = df[
        [
            "n_spikes_total",
            "sorted_unit_spikes_total_mclust",
            "raw_spike_events_total",
        ]
    ].max(axis=1)

    df["best_behavior_count"] = df[
        [
            "n_trials",
            "n_event_times_total",
            "n_position_samples",
        ]
    ].max(axis=1)

    # -----------------------------------------------------------------
    # Derived quantitative features
    # -----------------------------------------------------------------

    duration_min = safe_ratio(df["recording_duration_s"], 60)

    df["recording_duration_min"] = duration_min

    df["spikes_per_unit"] = safe_ratio(df["best_spike_count"], df["best_unit_count"])
    df["units_per_minute"] = safe_ratio(df["best_unit_count"], duration_min)
    df["spikes_per_minute"] = safe_ratio(df["best_spike_count"], duration_min)
    df["trials_per_minute"] = safe_ratio(df["n_trials"], duration_min)
    df["events_per_trial"] = safe_ratio(df["n_event_times_total"], df["n_trials"])
    df["lfp_channels_per_unit"] = safe_ratio(df["n_lfp_channels"], df["best_unit_count"])
    df["electrodes_per_unit"] = safe_ratio(df["n_electrodes"], df["best_unit_count"])

    # Log-transformed versions useful for PCA/ML.
    log_columns = [
        "best_unit_count",
        "best_spike_count",
        "n_lfp_channels",
        "n_electrodes",
        "n_trials",
        "n_event_times_total",
        "n_position_samples",
        "recording_duration_s",
        "spikes_per_unit",
        "spikes_per_minute",
    ]

    for column in log_columns:
        df["log_" + column] = np.log1p(df[column].clip(lower=0))

    # -----------------------------------------------------------------
    # Metadata-enriched reuse flags
    # -----------------------------------------------------------------

    df["can_do_spike_analysis"] = (
        df["extraction_success"]
        & df["has_spike_metadata"]
        & df["has_unit_metadata"]
        & (df["best_unit_count"] > 0)
    )

    df["can_do_lfp_analysis"] = (
        df["extraction_success"]
        & df["has_lfp_metadata"]
        & (df["n_lfp_channels"] > 0)
    )

    df["can_do_behavior_analysis"] = (
        df["extraction_success"]
        & (
            df["has_trial_metadata"]
            | df["has_event_metadata"]
            | df["has_position_metadata"]
        )
    )

    df["can_do_position_analysis"] = (
        df["extraction_success"]
        & df["has_position_metadata"]
    )

    df["can_do_spike_behavior_analysis"] = (
        df["can_do_spike_analysis"]
        & df["can_do_behavior_analysis"]
    )

    df["can_do_spike_position_analysis"] = (
        df["can_do_spike_analysis"]
        & df["can_do_position_analysis"]
    )

    df["can_do_lfp_behavior_analysis"] = (
        df["can_do_lfp_analysis"]
        & df["can_do_behavior_analysis"]
    )

    df["can_do_cross_dataset_comparison"] = (
        df["can_do_spike_analysis"]
        & (
            df["has_brain_region_metadata"]
            | df["has_standardized_format"]
            | df["brain_regions"].str.strip().ne("")
        )
    )

    # -----------------------------------------------------------------
    # Scores
    # -----------------------------------------------------------------

    reuse_score_components = [
        "can_do_spike_analysis",
        "can_do_lfp_analysis",
        "can_do_behavior_analysis",
        "can_do_position_analysis",
        "can_do_spike_behavior_analysis",
        "can_do_cross_dataset_comparison",
        "has_subject_metadata",
        "has_session_metadata",
        "has_session_date",
        "has_sampling_rate_metadata",
        "has_brain_region_metadata",
    ]

    df["cross_dataset_reuse_score"] = df[reuse_score_components].mean(axis=1)

    # More neural-data-oriented score. This downweights pure metadata.
    data_reuse_components = [
        "can_do_spike_analysis",
        "can_do_lfp_analysis",
        "can_do_behavior_analysis",
        "can_do_spike_behavior_analysis",
        "can_do_spike_position_analysis",
        "can_do_lfp_behavior_analysis",
    ]

    df["data_analysis_potential_score"] = df[data_reuse_components].mean(axis=1)

    df["missing_requirements"] = df.apply(make_missing_requirements, axis=1)
    df["recommended_analysis_type"] = df.apply(recommend_analysis_type, axis=1)

    # -----------------------------------------------------------------
    # Useful categorical simplifications for future AFC/ACM
    # -----------------------------------------------------------------

    context = df["behavioral_context"].str.lower()

    df["context_open_field"] = context.str.contains(
        "open field|navigation",
        regex=True,
        na=False,
    )

    df["context_object"] = context.str.contains(
        "object|obj",
        regex=True,
        na=False,
    )

    df["context_task"] = context.str.contains(
        "task|touchscreen|touchandsee|sequence|figure",
        regex=True,
        na=False,
    )

    df["context_sleep"] = context.str.contains(
        "sleep",
        regex=True,
        na=False,
    )

    df["recording_profile_group"] = np.select(
        [
            df["can_do_spike_behavior_analysis"],
            df["can_do_spike_position_analysis"],
            df["can_do_spike_analysis"] & df["can_do_lfp_analysis"],
            df["can_do_spike_analysis"],
            df["can_do_lfp_analysis"],
            df["can_do_behavior_analysis"],
        ],
        [
            "spike_behavior",
            "spike_position",
            "spike_lfp",
            "spike_only",
            "lfp_only",
            "behavior_only",
        ],
        default="metadata_only",
    )

    # -----------------------------------------------------------------
    # Output files
    # -----------------------------------------------------------------

    enriched_path = output_dir / "ca1_metadata_enriched_features.csv"
    reuse_path = output_dir / "ca1_reuse_recommendations.csv"
    availability_path = output_dir / "ca1_feature_availability_report.csv"

    df.to_csv(enriched_path, index=False)

    recommendation_columns = [
        "dataset_short_name",
        "session_id",
        "subject_id",
        "session_date",
        "behavioral_context",
        "recording_system",
        "brain_regions",
        "can_do_spike_analysis",
        "can_do_lfp_analysis",
        "can_do_behavior_analysis",
        "can_do_position_analysis",
        "can_do_spike_behavior_analysis",
        "can_do_spike_position_analysis",
        "can_do_lfp_behavior_analysis",
        "can_do_cross_dataset_comparison",
        "cross_dataset_reuse_score",
        "data_analysis_potential_score",
        "recommended_analysis_type",
        "missing_requirements",
    ]

    df[recommendation_columns].to_csv(reuse_path, index=False)

    grouped_rows = []

    for dataset_name, group in df.groupby("dataset_short_name"):
        row = {
            "dataset_short_name": dataset_name,
            "n_rows": len(group),
            "mean_cross_dataset_reuse_score": group["cross_dataset_reuse_score"].mean(),
            "mean_data_analysis_potential_score": group["data_analysis_potential_score"].mean(),
            "n_can_do_spike_analysis": int(group["can_do_spike_analysis"].sum()),
            "n_can_do_lfp_analysis": int(group["can_do_lfp_analysis"].sum()),
            "n_can_do_behavior_analysis": int(group["can_do_behavior_analysis"].sum()),
            "n_can_do_spike_behavior_analysis": int(group["can_do_spike_behavior_analysis"].sum()),
            "n_can_do_spike_position_analysis": int(group["can_do_spike_position_analysis"].sum()),
            "n_can_do_cross_dataset_comparison": int(group["can_do_cross_dataset_comparison"].sum()),
            "total_best_units": group["best_unit_count"].sum(),
            "total_best_spikes": group["best_spike_count"].sum(),
            "total_lfp_channels": group["n_lfp_channels"].sum(),
            "total_trials": group["n_trials"].sum(),
            "total_events": group["n_event_times_total"].sum(),
            "recording_profile_groups": "; ".join(
                sorted(group["recording_profile_group"].unique())
            ),
        }

        grouped_rows.append(row)

    availability = pd.DataFrame(grouped_rows)
    availability.to_csv(availability_path, index=False)

    print("Done.")
    print("Input:", input_csv)
    print("Enriched features:", enriched_path)
    print("Reuse recommendations:", reuse_path)
    print("Feature availability report:", availability_path)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build metadata-enriched features for cross-dataset CA1 analysis."
    )

    parser.add_argument(
        "harmonized_csv",
        help="Path to ca1_harmonized_sessions.csv",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: same folder as harmonized CSV.",
    )

    args = parser.parse_args()

    build_metadata_enriched_features(
        input_csv=args.harmonized_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()