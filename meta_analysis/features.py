# -*- coding: utf-8 -*-
"""Feature engineering and scoring for harmonized CA1 metadata rows.

This updated version explicitly integrates quantitative metadata / recording-level
variables into the meta-analysis, not only binary metadata availability flags.
"""

import numpy as np
import pandas as pd

FLAG_COLUMNS = [
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

NUMERIC_COLUMNS = [
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

TEXT_COLUMNS = [
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

COMPLETENESS_FLAGS = [
    "has_subject_metadata",
    "has_session_metadata",
    "has_session_date",
    "has_spike_metadata",
    "has_unit_metadata",
    "has_lfp_metadata",
    "has_trial_metadata",
    "has_event_metadata",
    "has_position_metadata",
    "has_sampling_rate_metadata",
    "has_brain_region_metadata",
]

EPHYS_FLAGS = [
    "has_spike_metadata",
    "has_unit_metadata",
    "has_sorted_unit_metadata",
    "has_lfp_metadata",
    "has_sampling_rate_metadata",
    "has_brain_region_metadata",
]

BEHAVIOR_FLAGS = [
    "has_trial_metadata",
    "has_event_metadata",
    "has_position_metadata",
]

OPENMINDS_FLAGS = [
    "has_subject_metadata",
    "has_session_metadata",
    "has_session_date",
    "has_brain_region_metadata",
    "has_sampling_rate_metadata",
    "has_standardized_format",
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

# Quantitative variables used in the updated PCA / ML meta-analysis.
# These are intentionally not dataset identifiers.
QUANTITATIVE_ML_FEATURES = [
    "metadata_completeness_score",
    "ephys_richness_score",
    "behavior_richness_score",
    "openminds_readiness_score",
    "cross_dataset_reuse_score",
    "data_analysis_potential_score",
    "log_best_unit_count",
    "log_best_spike_count",
    "log_best_behavior_count",
    "log_n_lfp_channels",
    "log_n_electrodes",
    "log_n_trials",
    "log_n_event_times_total",
    "log_n_position_samples",
    "log_recording_duration_s",
    "log_spikes_per_unit",
    "log_units_per_minute",
    "log_spikes_per_minute",
    "log_trials_per_minute",
    "log_events_per_trial",
    "log_lfp_channels_per_unit",
    "log_electrodes_per_unit",
]

# Categorical metadata used for AFC/ACM-like analysis.
#
# Important:
# We intentionally EXCLUDE dataset_short_name, source_type, source_format,
# recording_system and raw behavioral_context from the default AFC/ACM analysis.
# Otherwise the categorical analysis mostly rediscovers dataset/file-format
# identity instead of showing reusable metadata profiles.
CATEGORICAL_FEATURES = [
    "metadata_profile_label",
    "recommended_analysis_type",
    "recording_profile_group",

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

    "can_do_spike_analysis",
    "can_do_lfp_analysis",
    "can_do_behavior_analysis",
    "can_do_position_analysis",
    "can_do_spike_behavior_analysis",
    "can_do_spike_position_analysis",
    "can_do_lfp_behavior_analysis",
    "can_do_cross_dataset_comparison",

    "context_open_field",
    "context_object",
    "context_task",
    "context_sleep",
]


def _to_bool(series):
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def _safe_ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    if isinstance(denominator, pd.Series):
        denominator = pd.to_numeric(denominator, errors="coerce")
    else:
        denominator = pd.Series(denominator, index=numerator.index)
        denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = numerator.fillna(0)
    denominator = denominator.fillna(0)
    values = np.where(denominator > 0, numerator / denominator, 0)
    return pd.Series(values, index=numerator.index).replace([np.inf, -np.inf], 0).fillna(0)


def prepare_dataframe(rows):
    """Create the harmonized session table and all engineered features."""
    df = pd.DataFrame(rows)

    for col in FLAG_COLUMNS:
        if col not in df.columns:
            df[col] = False
        df[col] = _to_bool(df[col])

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Classical metadata quality scores.
    df["metadata_completeness_score"] = df[COMPLETENESS_FLAGS].mean(axis=1)
    df["ephys_richness_score"] = df[EPHYS_FLAGS].mean(axis=1)
    df["behavior_richness_score"] = df[BEHAVIOR_FLAGS].mean(axis=1)
    df["openminds_readiness_score"] = df[OPENMINDS_FLAGS].mean(axis=1)

    # Best available quantitative variables across heterogeneous formats.
    df["best_unit_count"] = df[["n_units", "n_spiketrains", "sorted_units_mclust"]].max(axis=1)
    df["best_spike_count"] = df[["n_spikes_total", "sorted_unit_spikes_total_mclust", "raw_spike_events_total"]].max(axis=1)
    df["best_behavior_count"] = df[["n_trials", "n_event_times_total", "n_position_samples"]].max(axis=1)

    df["recording_duration_min"] = _safe_ratio(df["recording_duration_s"], 60)
    df["spikes_per_unit"] = _safe_ratio(df["best_spike_count"], df["best_unit_count"])
    df["units_per_minute"] = _safe_ratio(df["best_unit_count"], df["recording_duration_min"])
    df["spikes_per_minute"] = _safe_ratio(df["best_spike_count"], df["recording_duration_min"])
    df["trials_per_minute"] = _safe_ratio(df["n_trials"], df["recording_duration_min"])
    df["events_per_trial"] = _safe_ratio(df["n_event_times_total"], df["n_trials"])
    df["lfp_channels_per_unit"] = _safe_ratio(df["n_lfp_channels"], df["best_unit_count"])
    df["electrodes_per_unit"] = _safe_ratio(df["n_electrodes"], df["best_unit_count"])

    # Log-scaled quantities for PCA/ML. This helps reduce dominance of very large counts.
    log_sources = [
        "n_units", "n_spikes_total", "raw_spike_events_total", "n_lfp_channels",
        "n_electrodes", "n_trials", "n_event_times_total", "n_position_samples",
        "recording_duration_s", "best_unit_count", "best_spike_count",
        "best_behavior_count", "spikes_per_unit", "units_per_minute",
        "spikes_per_minute", "trials_per_minute", "events_per_trial",
        "lfp_channels_per_unit", "electrodes_per_unit",
    ]
    for col in log_sources:
        df["log_" + col] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0))

    # Reuse flags: these give a concrete interpretation of the metadata-enriched table.
    df["can_do_spike_analysis"] = (
        df["extraction_success"] & df["has_spike_metadata"] & df["has_unit_metadata"] & (df["best_unit_count"] > 0)
    )
    df["can_do_lfp_analysis"] = df["extraction_success"] & df["has_lfp_metadata"] & (df["n_lfp_channels"] > 0)
    df["can_do_behavior_analysis"] = df["extraction_success"] & (
        df["has_trial_metadata"] | df["has_event_metadata"] | df["has_position_metadata"]
    )
    df["can_do_position_analysis"] = df["extraction_success"] & df["has_position_metadata"]
    df["can_do_spike_behavior_analysis"] = df["can_do_spike_analysis"] & df["can_do_behavior_analysis"]
    df["can_do_spike_position_analysis"] = df["can_do_spike_analysis"] & df["can_do_position_analysis"]
    df["can_do_lfp_behavior_analysis"] = df["can_do_lfp_analysis"] & df["can_do_behavior_analysis"]
    df["can_do_cross_dataset_comparison"] = df["can_do_spike_analysis"] & (
        df["has_brain_region_metadata"] | df["has_standardized_format"] | df["brain_regions"].str.strip().ne("")
    )

    df["cross_dataset_reuse_score"] = df[[
        "can_do_spike_analysis", "can_do_lfp_analysis", "can_do_behavior_analysis",
        "can_do_position_analysis", "can_do_spike_behavior_analysis", "can_do_cross_dataset_comparison",
        "has_subject_metadata", "has_session_metadata", "has_session_date",
        "has_sampling_rate_metadata", "has_brain_region_metadata",
    ]].mean(axis=1)

    df["data_analysis_potential_score"] = df[[
        "can_do_spike_analysis", "can_do_lfp_analysis", "can_do_behavior_analysis",
        "can_do_spike_behavior_analysis", "can_do_spike_position_analysis", "can_do_lfp_behavior_analysis",
    ]].mean(axis=1)

    context = df["behavioral_context"].str.lower()
    df["context_open_field"] = context.str.contains("open field|navigation", regex=True, na=False)
    df["context_object"] = context.str.contains("object|obj", regex=True, na=False)
    df["context_task"] = context.str.contains("task|touchscreen|touchandsee|sequence|figure", regex=True, na=False)
    df["context_sleep"] = context.str.contains("sleep", regex=True, na=False)

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
            "spike_behavior", "spike_position", "spike_lfp",
            "spike_only", "lfp_only", "behavior_only",
        ],
        default="metadata_only",
    )

    df["missing_requirements"] = df.apply(make_missing_requirements, axis=1)
    df["recommended_analysis_type"] = df.apply(recommend_analysis_type, axis=1)
    df["metadata_profile_label"] = df.apply(assign_profile_label, axis=1)

    return df


def make_missing_requirements(row):
    missing = []
    if not row.get("has_subject_metadata", False): missing.append("subject metadata")
    if not row.get("has_session_metadata", False): missing.append("session metadata")
    if not row.get("has_session_date", False): missing.append("session date")
    if not row.get("has_spike_metadata", False): missing.append("spike/unit metadata")
    if not row.get("has_lfp_metadata", False): missing.append("LFP metadata")
    if not row.get("has_trial_metadata", False) and not row.get("has_event_metadata", False) and not row.get("has_position_metadata", False):
        missing.append("behavioral metadata")
    if not row.get("has_sampling_rate_metadata", False): missing.append("sampling-rate metadata")
    if not row.get("has_brain_region_metadata", False): missing.append("brain-region metadata")
    return "none" if not missing else "; ".join(missing)


def recommend_analysis_type(row):
    if row.get("can_do_spike_behavior_analysis", False): return "spike-behavior analysis"
    if row.get("can_do_spike_position_analysis", False): return "spike-position / navigation analysis"
    if row.get("can_do_lfp_behavior_analysis", False): return "LFP-behavior analysis"
    if row.get("can_do_spike_analysis", False): return "spike/unit analysis"
    if row.get("can_do_lfp_analysis", False): return "LFP metadata-level analysis"
    if row.get("can_do_behavior_analysis", False): return "behavior-only metadata analysis"
    return "metadata curation only"


def assign_profile_label(row):
    if not row.get("extraction_success", False):
        return "metadata-limited / extraction failed"
    if row.get("can_do_spike_behavior_analysis", False):
        return "spike+behavior reusable profile"
    if row.get("can_do_spike_position_analysis", False):
        return "spike+position navigation profile"
    if row.get("has_standardized_format", False) and row.get("has_unit_metadata", False):
        return "standardized NWB unit-rich profile"
    if row.get("has_lfp_metadata", False) and row.get("has_unit_metadata", False):
        return "spike+LFP ephys profile"
    if row.get("has_unit_metadata", False):
        return "unit-only ephys profile"
    if row.get("has_lfp_metadata", False):
        return "LFP-only ephys profile"
    return "metadata-limited profile"
