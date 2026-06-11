# -*- coding: utf-8 -*-
"""Feature engineering and scoring for harmonized metadata rows."""

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


def prepare_dataframe(rows):
    df = pd.DataFrame(rows)

    for col in FLAG_COLUMNS:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["dataset_short_name", "dataset_name", "source_type", "source_format", "session_id", "subject_id", "session_date", "recording_system", "behavioral_context", "brain_regions", "unit_source", "error"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["metadata_completeness_score"] = df[COMPLETENESS_FLAGS].mean(axis=1)
    df["ephys_richness_score"] = df[EPHYS_FLAGS].mean(axis=1)
    df["behavior_richness_score"] = df[BEHAVIOR_FLAGS].mean(axis=1)
    df["openminds_readiness_score"] = df[OPENMINDS_FLAGS].mean(axis=1)

    # Additional interpretable derived ratios.
    df["log_n_units"] = np.log1p(df["n_units"])
    df["log_n_spikes_total"] = np.log1p(df["n_spikes_total"])
    df["log_raw_spike_events_total"] = np.log1p(df["raw_spike_events_total"])
    df["log_n_lfp_channels"] = np.log1p(df["n_lfp_channels"])
    df["log_n_electrodes"] = np.log1p(df["n_electrodes"])
    df["log_n_trials"] = np.log1p(df["n_trials"])
    df["log_n_event_times_total"] = np.log1p(df["n_event_times_total"])

    df["metadata_profile_label"] = df.apply(assign_profile_label, axis=1)

    return df


def assign_profile_label(row):
    if not row.get("extraction_success", False):
        return "metadata-limited / extraction failed"

    has_position = row.get("has_position_metadata", False)
    has_lfp = row.get("has_lfp_metadata", False)
    has_trials = row.get("has_trial_metadata", False)
    has_events = row.get("has_event_metadata", False)
    has_units = row.get("has_unit_metadata", False)
    standardized = row.get("has_standardized_format", False)

    if standardized and has_units:
        return "standardized NWB unit-rich profile"

    if has_position and has_lfp and has_units:
        return "navigation/open-field ephys profile"

    if has_trials and has_events and has_units:
        return "task/event/unit-rich profile"

    if has_lfp and has_units:
        return "spike+LFP ephys profile"

    if has_units:
        return "unit-only ephys profile"

    return "metadata-limited profile"
