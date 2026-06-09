# -*- coding: utf-8 -*-
"""
CA1 metadata ML meta-analysis pipeline.

Goal:
    Combine extracted metadata JSON files from heterogeneous CA1-related
    electrophysiology datasets into one harmonized session-level table, then run
    a small metadata-based ML case study:
        - metadata completeness/reuse-readiness scores
        - missing metadata analysis
        - PCA of recording/session metadata profiles
        - unsupervised clustering

Expected inputs:
    1. Openfield Axona/Neuralynx JSON
    2. NWB extraction JSON
    3. Legacy touchscreen Neo/NIX JSON
    4. TouchAndSee extraction JSON

Outputs:
    meta_analysis/outputs/
        ca1_harmonized_sessions.csv
        ca1_dataset_summary.csv
        ca1_metadata_availability_by_dataset.csv
        ca1_missing_metadata_report.csv
        ca1_pca_coordinates.csv
        ca1_cluster_profiles.csv
        figures/*.png

Run example:
    python meta_analysis/run_ca1_metadata_meta_analysis.py ^
        outputs/extracted_metadata/d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json ^
        outputs/extracted_metadata/d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json ^
        outputs/extracted_metadata/d-d406a98c-ae5c-4fb3-9f0c-4cf4de9b1094-hbp-data-002061-data_legacy_touchscreen_metadata.json ^
        outputs/extracted_metadata/p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_extracted_metadata.json
"""

import argparse
import json
import math
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def safe_get(mapping, key, default=None):
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return default


def as_bool(value):
    return bool(value) if value is not None else False


def to_number(value, default=0):
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return default
            return value
        return float(value)
    except Exception:
        return default


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def list_to_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(str(k) for k in value.keys())
    return str(value)


def infer_dataset_short_name(dataset_name):
    name = str(dataset_name)

    if "40faae41" in name:
        return "openfield_ca1"

    if "885b4936" in name:
        return "nwb_ca1"

    if "d406a98c" in name or "002061" in name:
        return "legacy_touchscreen"

    if "p25b4e" in name or "01681" in name or "Pennartz" in name:
        return "touchandsee"

    return name[:40]


def infer_source_type(dataset):
    extractor = str(dataset.get("extractor", "")).lower()
    dataset_name = str(dataset.get("dataset_name", "")).lower()

    if "sessions" in dataset and "openfield" in extractor:
        return "openfield"

    if "sessions" in dataset and "legacy_touchscreen" in extractor:
        return "legacy_touchscreen"

    # New TouchAndSee internal Neo pickle extractor output
    if "files" in dataset and ("touchandsee" in extractor or "01681" in dataset_name or "p25b4e" in dataset_name):
        return "regular_pickle_or_touchandsee"

    if "files" in dataset:
        extensions = dataset.get("extensions_searched", [])
        if ".nwb" in extensions:
            return "nwb"
        if ".pkl" in extensions:
            return "regular_pickle_or_touchandsee"

    return "unknown"


def has_any_text(value):
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return len(value) > 0
    return str(value).strip() != ""


def sum_dict_values(d):
    if not isinstance(d, dict):
        return 0
    total = 0
    for value in d.values():
        total += to_number(value, default=0)
    return total


# ---------------------------------------------------------------------
# Dataset-specific harmonizers
# ---------------------------------------------------------------------


def harmonize_openfield(dataset):
    rows = []
    dataset_name = dataset.get("dataset_name")
    dataset_short_name = infer_dataset_short_name(dataset_name)

    for session in dataset.get("sessions", []):
        summary = session.get("summary") or {}
        file_counts = session.get("file_counts") or {}
        neuralynx = session.get("neuralynx_metadata") or {}
        axona = session.get("axona_metadata") or {}
        mclust = session.get("mclust_metadata") or {}
        optitrack = session.get("optitrack_metadata") or {}

        recording_system = session.get("recording_system") or ""

        lfp_sampling_rates = []
        if neuralynx.get("lfp_sampling_rates_hz"):
            lfp_sampling_rates.extend(neuralynx.get("lfp_sampling_rates_hz") or [])
        if axona.get("eeg_sample_rates_hz"):
            lfp_sampling_rates.extend(axona.get("eeg_sample_rates_hz") or [])
        if axona.get("eeg_sampling_rates_hz"):
            lfp_sampling_rates.extend(axona.get("eeg_sampling_rates_hz") or [])

        n_position_samples = (
            axona.get("n_position_samples")
            or neuralynx.get("n_position_records")
            or optitrack.get("n_timestamps")
            or optitrack.get("n_position_samples")
        )

        # The openfield extractor has existed in two naming variants.
        # Support both to avoid silently zeroing useful metadata.
        n_units_best = (
            summary.get("n_units_best_available")
            or summary.get("n_units")
            or mclust.get("sorted_units_mclust")
            or mclust.get("n_units")
        )
        n_spikes_best = (
            summary.get("sorted_unit_spikes_total_mclust")
            or mclust.get("sorted_unit_spikes_total_mclust")
            or mclust.get("n_spikes_total")
            or summary.get("n_spikes_total")
        )
        raw_spike_events = (
            summary.get("raw_spike_events_total")
            or neuralynx.get("raw_spike_events_total")
            or neuralynx.get("n_spike_events_total")
            or axona.get("n_spikes_total")
            or summary.get("n_spikes_total")
        )
        sorted_units = (
            summary.get("sorted_units_mclust")
            or mclust.get("sorted_units_mclust")
            or mclust.get("n_units")
        )
        sorted_unit_spikes = (
            summary.get("sorted_unit_spikes_total_mclust")
            or mclust.get("sorted_unit_spikes_total_mclust")
            or mclust.get("n_spikes_total")
        )

        row = {
            "dataset_short_name": dataset_short_name,
            "dataset_name": dataset_name,
            "source_type": "openfield",
            "source_format": "Axona/Neuralynx/MClust/OptiTrack",
            "session_id": session.get("session_id"),
            "subject_id": session.get("subject_id"),
            "session_date": session.get("session_date"),
            "recording_system": recording_system,
            "behavioral_context": "open field / navigation",
            "extraction_success": True,

            "n_files": file_counts.get("n_files_total"),
            "file_extensions": list_to_text(file_counts.get("extensions")),

            "recording_duration_s": summary.get("recording_duration_s"),
            "n_units": n_units_best,
            "n_spiketrains": n_units_best,
            "n_spikes_total": n_spikes_best,
            "raw_spike_events_total": raw_spike_events,
            "sorted_units_mclust": sorted_units,
            "sorted_unit_spikes_total_mclust": sorted_unit_spikes,
            "n_lfp_channels": summary.get("n_lfp_channels"),
            "n_electrodes": None,
            "n_trials": 0,
            "n_event_objects": None,
            "n_event_times_total": neuralynx.get("n_event_records"),
            "n_position_samples": n_position_samples,

            "sampling_rates_hz": list_to_text(lfp_sampling_rates),
            "brain_regions": "",
            "unit_source": summary.get("n_units_source"),

            "has_subject_metadata": has_any_text(session.get("subject_id")),
            "has_session_metadata": has_any_text(session.get("session_id")),
            "has_session_date": has_any_text(session.get("session_date")),
            "has_recording_duration": summary.get("recording_duration_s") is not None,
            "has_spike_metadata": as_bool(summary.get("has_raw_spike_metadata")) or as_bool(summary.get("has_sorted_unit_metadata")) or as_bool(summary.get("has_spike_metadata")) or to_number(raw_spike_events, 0) > 0,
            "has_unit_metadata": as_bool(summary.get("has_sorted_unit_metadata")) or as_bool(summary.get("has_unit_metadata")) or to_number(n_units_best, 0) > 0,
            "has_sorted_unit_metadata": as_bool(summary.get("has_sorted_unit_metadata")) or as_bool(summary.get("has_unit_metadata")) or to_number(n_units_best, 0) > 0,
            "has_lfp_metadata": as_bool(summary.get("has_lfp_metadata")),
            "has_trial_metadata": False,
            "has_event_metadata": neuralynx.get("events_success") is True,
            "has_position_metadata": as_bool(summary.get("has_position_metadata")),
            "has_sampling_rate_metadata": len(lfp_sampling_rates) > 0,
            "has_brain_region_metadata": False,
            "has_standardized_format": False,
            "has_openminds_candidate_metadata": True,

            "error": "",
        }

        rows.append(row)

    return rows


def harmonize_nwb(dataset):
    rows = []
    dataset_name = dataset.get("dataset_name")
    dataset_short_name = infer_dataset_short_name(dataset_name)

    for item in dataset.get("files", []):
        file_meta = item.get("file_metadata") or {}
        nwb = item.get("nwb_extraction") or {}

        subject = nwb.get("subject") or {}

        success = nwb.get("success") is True

        electrode_locations = nwb.get("electrode_locations") or []
        processing_modules = nwb.get("processing_modules") or []
        intervals = nwb.get("intervals") or []
        unit_columns = nwb.get("unit_columns") or []
        electrode_columns = nwb.get("electrode_columns") or []

        session_description = nwb.get("session_description") or ""
        behavioral_context = session_description
        if not behavioral_context:
            behavioral_context = "NWB electrophysiology"

        row = {
            "dataset_short_name": dataset_short_name,
            "dataset_name": dataset_name,
            "source_type": "nwb",
            "source_format": "NWB",
            "session_id": nwb.get("identifier") or file_meta.get("session_id") or Path(file_meta.get("file_name", "")).stem,
            "subject_id": subject.get("subject_id") or file_meta.get("subject_id") or file_meta.get("animal_id"),
            "session_date": str(nwb.get("session_start_time") or "")[:10],
            "recording_system": list_to_text(nwb.get("devices")),
            "behavioral_context": behavioral_context,
            "extraction_success": success,

            "n_files": 1,
            "file_extensions": file_meta.get("file_extension"),

            "recording_duration_s": None,
            "n_units": nwb.get("n_units"),
            "n_spiketrains": nwb.get("n_units"),
            "n_spikes_total": None,
            "raw_spike_events_total": None,
            "sorted_units_mclust": None,
            "sorted_unit_spikes_total_mclust": None,
            "n_lfp_channels": None,
            "n_electrodes": nwb.get("n_electrodes"),
            "n_trials": nwb.get("n_trials"),
            "n_event_objects": len(intervals),
            "n_event_times_total": None,
            "n_position_samples": None,

            "sampling_rates_hz": "unit sampling_rate column" if "sampling_rate" in unit_columns else "",
            "brain_regions": list_to_text(electrode_locations),
            "unit_source": "NWB units table",

            "has_subject_metadata": has_any_text(subject.get("subject_id")),
            "has_session_metadata": has_any_text(nwb.get("identifier")),
            "has_session_date": has_any_text(nwb.get("session_start_time")),
            "has_recording_duration": False,
            "has_spike_metadata": to_number(nwb.get("n_units"), 0) > 0,
            "has_unit_metadata": to_number(nwb.get("n_units"), 0) > 0,
            "has_sorted_unit_metadata": to_number(nwb.get("n_units"), 0) > 0,
            "has_lfp_metadata": False,
            "has_trial_metadata": to_number(nwb.get("n_trials"), 0) > 0,
            "has_event_metadata": len(intervals) > 0,
            "has_position_metadata": "behavior" in processing_modules,
            "has_sampling_rate_metadata": "sampling_rate" in unit_columns,
            "has_brain_region_metadata": len(electrode_locations) > 0,
            "has_standardized_format": True,
            "has_openminds_candidate_metadata": True,

            "species": subject.get("species"),
            "sex": subject.get("sex"),
            "institution": nwb.get("institution"),
            "experimenter": list_to_text(nwb.get("experimenter")),
            "electrode_columns": list_to_text(electrode_columns),
            "unit_columns": list_to_text(unit_columns),
            "error": "" if success else normalize_text(nwb.get("error")),
        }

        rows.append(row)

    return rows


def harmonize_legacy_touchscreen(dataset):
    rows = []
    dataset_name = dataset.get("dataset_name")
    dataset_short_name = infer_dataset_short_name(dataset_name)

    for session in dataset.get("sessions", []):
        summary = session.get("summary") or {}

        event_counts = summary.get("event_counts") or {}
        event_total = sum_dict_values(event_counts)

        unit_categories = summary.get("unit_categories") or {}
        lfp_categories = summary.get("lfp_categories") or {}
        trial_categories = summary.get("trial_categories") or {}

        regions = []
        for key in ["area", "recording_group", "tetrode_area"]:
            if key in unit_categories:
                regions.extend(unit_categories[key].get("values", []))
        if "recording_group" in lfp_categories:
            regions.extend(lfp_categories["recording_group"].get("values", []))
        regions = sorted(set(str(x) for x in regions if str(x).strip() and str(x) != "no_data"))

        row = {
            "dataset_short_name": dataset_short_name,
            "dataset_name": dataset_name,
            "source_type": "legacy_touchscreen",
            "source_format": "legacy Neo/NIX",
            "session_id": session.get("session_id"),
            "subject_id": session.get("animal_id"),
            "session_date": session.get("session_date"),
            "recording_system": "legacy Neo/NIX",
            "behavioral_context": "touchscreen / tactile-visual task",
            "extraction_success": session.get("success") is True,

            "n_files": None,
            "file_extensions": ".pkl; .nio",

            "recording_duration_s": None,
            "n_units": summary.get("n_units_from_dataframe"),
            "n_spiketrains": summary.get("n_spiketrains"),
            "n_spikes_total": summary.get("n_spikes_total"),
            "raw_spike_events_total": None,
            "sorted_units_mclust": None,
            "sorted_unit_spikes_total_mclust": None,
            "n_lfp_channels": summary.get("n_lfp_channels_from_dataframe"),
            "n_electrodes": None,
            "n_trials": summary.get("n_trials_from_dataframe"),
            "n_event_objects": summary.get("n_events"),
            "n_event_times_total": event_total,
            "n_position_samples": None,

            "sampling_rates_hz": "",
            "brain_regions": list_to_text(regions),
            "unit_source": "Neo spiketrains / unit dataframe",

            "has_subject_metadata": has_any_text(session.get("animal_id")),
            "has_session_metadata": has_any_text(session.get("session_id")),
            "has_session_date": has_any_text(session.get("session_date")),
            "has_recording_duration": False,
            "has_spike_metadata": as_bool(summary.get("has_spike_metadata")),
            "has_unit_metadata": to_number(summary.get("n_units_from_dataframe"), 0) > 0,
            "has_sorted_unit_metadata": to_number(summary.get("n_units_from_dataframe"), 0) > 0,
            "has_lfp_metadata": as_bool(summary.get("has_lfp_metadata")),
            "has_trial_metadata": as_bool(summary.get("has_trial_metadata")),
            "has_event_metadata": as_bool(summary.get("has_event_metadata")),
            "has_position_metadata": (
                "has_nosetracking" in trial_categories or "has_whiskertracking" in trial_categories
            ),
            "has_sampling_rate_metadata": False,
            "has_brain_region_metadata": len(regions) > 0,
            "has_standardized_format": False,
            "has_openminds_candidate_metadata": True,

            "error": "" if session.get("success") is True else normalize_text(session.get("error")),
        }

        rows.append(row)

    return rows


def harmonize_regular_pickle_or_touchandsee(dataset):
    """
    Handles the regular extraction JSON for TouchAndSee.

    Important:
        In the currently provided file, pickle/Neo extraction failed for all .pkl
        files because of old Neo pickle compatibility. We still include these
        sessions in the harmonized table as metadata-limited rows, using file
        metadata only. Once the pickle compatibility is fixed and this JSON is
        regenerated, this function can be extended to read object_summary.
    """

    rows = []
    dataset_name = dataset.get("dataset_name")
    dataset_short_name = infer_dataset_short_name(dataset_name)

    for item in dataset.get("files", []):
        file_meta = item.get("file_metadata") or {}
        pickle_info = item.get("pickle_extraction") or {}
        neo_info = item.get("neo_extraction") or {}
        touchandsee_info = item.get("touchandsee_extraction") or {}

        ext = file_meta.get("file_extension")

        # Keep session-like data files as rows. Ignore code/descriptor files.
        if ext not in [".pkl", ".nwb"]:
            continue

        pickle_success = pickle_info.get("success") is True
        neo_success = neo_info.get("success") is True
        touchandsee_success = touchandsee_info.get("success") is True
        success = pickle_success or neo_success or touchandsee_success

        object_summary = (
            touchandsee_info.get("object_summary")
            or pickle_info.get("object_summary")
            or neo_info.get("object_summary")
            or {}
        )

        row = {
            "dataset_short_name": dataset_short_name,
            "dataset_name": dataset_name,
            "source_type": "touchandsee_pickle",
            "source_format": "Neo pickle",
            "session_id": file_meta.get("session_id") or Path(file_meta.get("file_name", "")).stem,
            "subject_id": file_meta.get("subject_id") or file_meta.get("animal_id"),
            "session_date": file_meta.get("session_date"),
            "recording_system": "Neo pickle",
            "behavioral_context": file_meta.get("task_label") or "TouchAndSee",
            "extraction_success": success,

            "n_files": 1,
            "file_extensions": ext,

            # If object_summary exists in future, these can be populated.
            "recording_duration_s": object_summary.get("duration_s"),
            "n_units": object_summary.get("n_units"),
            "n_spiketrains": object_summary.get("n_spiketrains"),
            "n_spikes_total": object_summary.get("n_spikes_total"),
            "raw_spike_events_total": None,
            "sorted_units_mclust": None,
            "sorted_unit_spikes_total_mclust": None,
            "n_lfp_channels": object_summary.get("n_lfp_channels") or object_summary.get("n_analogsignals"),
            "n_electrodes": object_summary.get("n_electrodes"),
            "n_trials": object_summary.get("n_trials") or object_summary.get("n_segments"),
            "n_event_objects": object_summary.get("n_events"),
            "n_event_times_total": object_summary.get("n_event_times_total"),
            "n_position_samples": object_summary.get("n_position_samples"),

            "sampling_rates_hz": list_to_text(object_summary.get("sampling_rates_hz")),
            "brain_regions": list_to_text(object_summary.get("brain_regions")),
            "unit_source": "Neo pickle" if success else "not loaded",

            "has_subject_metadata": has_any_text(file_meta.get("subject_id") or file_meta.get("animal_id")),
            "has_session_metadata": has_any_text(file_meta.get("session_id")),
            "has_session_date": has_any_text(file_meta.get("session_date")),
            "has_recording_duration": object_summary.get("duration_s") is not None,
            "has_spike_metadata": to_number(object_summary.get("n_spikes_total"), 0) > 0,
            "has_unit_metadata": to_number(object_summary.get("n_units"), 0) > 0 or to_number(object_summary.get("n_spiketrains"), 0) > 0,
            "has_sorted_unit_metadata": to_number(object_summary.get("n_units"), 0) > 0,
            "has_lfp_metadata": to_number(object_summary.get("n_lfp_channels") or object_summary.get("n_analogsignals"), 0) > 0,
            "has_trial_metadata": to_number(object_summary.get("n_trials") or object_summary.get("n_segments"), 0) > 0,
            "has_event_metadata": to_number(object_summary.get("n_events"), 0) > 0,
            "has_position_metadata": to_number(object_summary.get("n_position_samples"), 0) > 0,
            "has_sampling_rate_metadata": has_any_text(object_summary.get("sampling_rates_hz")),
            "has_brain_region_metadata": has_any_text(object_summary.get("brain_regions")),
            "has_standardized_format": False,
            "has_openminds_candidate_metadata": True,

            "file_size_mb": file_meta.get("file_size_mb"),
            "error": "" if success else normalize_text(pickle_info.get("error") or neo_info.get("error")),
        }

        rows.append(row)

    return rows


def harmonize_dataset(dataset):
    source_type = infer_source_type(dataset)

    if source_type == "openfield":
        return harmonize_openfield(dataset)

    if source_type == "nwb":
        return harmonize_nwb(dataset)

    if source_type == "legacy_touchscreen":
        return harmonize_legacy_touchscreen(dataset)

    if source_type == "regular_pickle_or_touchandsee":
        return harmonize_regular_pickle_or_touchandsee(dataset)

    raise ValueError("Could not infer dataset source type for: {}".format(dataset.get("dataset_name")))


# ---------------------------------------------------------------------
# Feature engineering and scoring
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# ML analysis
# ---------------------------------------------------------------------


def build_ml_matrix(df):
    """
    Build numeric feature matrix for PCA/clustering.
    """

    feature_columns = [
        # Scores
        "metadata_completeness_score",
        "ephys_richness_score",
        "behavior_richness_score",
        "openminds_readiness_score",

        # Log-scaled quantities
        "log_n_units",
        "log_n_spikes_total",
        "log_raw_spike_events_total",
        "log_n_lfp_channels",
        "log_n_electrodes",
        "log_n_trials",
        "log_n_event_times_total",

        # Flags
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
    ]

    X = df[feature_columns].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)

    return X, feature_columns


def standardize_matrix(X):
    values = X.values.astype(float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    scaled = (values - mean) / std
    return scaled, mean, std


def run_pca_and_clustering(df, n_clusters=4):
    """
    Run PCA and KMeans clustering.

    Requires scikit-learn. If unavailable, the script still produces the
    harmonized CSVs and reports, but skips ML outputs.
    """

    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as error:
        print("scikit-learn is not available. Skipping PCA/clustering.")
        print("Install with: python -m pip install scikit-learn")
        print("Error:", repr(error))
        return df, None, None, None

    X, feature_columns = build_ml_matrix(df)
    X_scaled, _, _ = standardize_matrix(X)

    n_samples = X_scaled.shape[0]
    n_components = 2 if n_samples >= 2 else 1

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    df_ml = df.copy()
    df_ml["pca_1"] = coords[:, 0]
    df_ml["pca_2"] = coords[:, 1] if n_components > 1 else 0.0

    # Ensure sensible cluster number.
    n_clusters = int(min(n_clusters, max(2, n_samples - 1)))
    if n_samples < 3:
        df_ml["cluster"] = 0
        return df_ml, pca, feature_columns, None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    df_ml["cluster"] = labels

    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = None

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("KMeans n_clusters:", n_clusters)
    print("Silhouette score:", sil)

    return df_ml, pca, feature_columns, sil


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


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


def plot_completeness_by_dataset(dataset_summary, figures_dir):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(dataset_summary))
    y = dataset_summary["mean_metadata_completeness_score"].values
    labels = dataset_summary["dataset_short_name"].values

    plt.bar(x, y)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean metadata completeness score")
    plt.title("Metadata completeness by dataset")
    plt.tight_layout()
    plt.savefig(figures_dir / "01_completeness_by_dataset.png", dpi=200)
    plt.close()


def plot_missingness_heatmap(df, figures_dir):
    availability = df.groupby("dataset_short_name")[COMPLETENESS_FLAGS].mean()

    plt.figure(figsize=(12, 5))
    plt.imshow(availability.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Fraction present")
    plt.yticks(np.arange(len(availability.index)), availability.index)
    plt.xticks(np.arange(len(availability.columns)), availability.columns, rotation=45, ha="right")
    plt.title("Metadata availability heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_metadata_availability_heatmap.png", dpi=200)
    plt.close()


def plot_pca(df, figures_dir):
    if "pca_1" not in df.columns:
        return

    plt.figure(figsize=(8, 6))

    datasets = sorted(df["dataset_short_name"].unique())
    for dataset in datasets:
        subset = df[df["dataset_short_name"] == dataset]
        plt.scatter(subset["pca_1"], subset["pca_2"], label=dataset, alpha=0.8)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA of CA1 metadata profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "03_pca_metadata_profiles.png", dpi=200)
    plt.close()


def plot_clusters(df, figures_dir):
    if "cluster" not in df.columns:
        return

    counts = df.groupby(["cluster", "dataset_short_name"]).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(counts))

    x = np.arange(len(counts.index))
    for col in counts.columns:
        values = counts[col].values
        plt.bar(x, values, bottom=bottom, label=col)
        bottom += values

    plt.xticks(x, [str(c) for c in counts.index])
    plt.xlabel("Cluster")
    plt.ylabel("Number of sessions/files")
    plt.title("Cluster composition by dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "04_cluster_composition.png", dpi=200)
    plt.close()


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
        "The TouchAndSee dataset is currently represented mainly by file/session metadata, "
        "because Neo/pickle loading failed in the provided extraction JSON. Once the legacy "
        "pickle compatibility issue is fixed and the extraction is rerun, the same pipeline "
        "will include its internal electrophysiology metadata automatically if present in the JSON."
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(input_paths, output_dir, n_clusters=4):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for path in input_paths:
        path = Path(path)
        print("Loading:", path)
        dataset = load_json(path)
        rows = harmonize_dataset(dataset)
        print("  rows:", len(rows), "source:", infer_source_type(dataset), "dataset:", dataset.get("dataset_name"))
        all_rows.extend(rows)

    df = prepare_dataframe(all_rows)

    # Run ML.
    df_ml, pca, feature_columns, silhouette = run_pca_and_clustering(df, n_clusters=n_clusters)

    dataset_summary = make_dataset_summary(df_ml)
    availability = make_availability_by_dataset(df_ml)
    missing_report = make_missing_metadata_report(df_ml)
    cluster_profiles = make_cluster_profiles(df_ml)

    # Save outputs.
    df_ml.to_csv(output_dir / "ca1_harmonized_sessions.csv", index=False)
    dataset_summary.to_csv(output_dir / "ca1_dataset_summary.csv", index=False)
    availability.to_csv(output_dir / "ca1_metadata_availability_by_dataset.csv", index=False)
    missing_report.to_csv(output_dir / "ca1_missing_metadata_report.csv", index=False)

    if "pca_1" in df_ml.columns:
        pca_cols = [
            "dataset_short_name",
            "session_id",
            "subject_id",
            "metadata_profile_label",
            "cluster",
            "pca_1",
            "pca_2",
        ]
        df_ml[pca_cols].to_csv(output_dir / "ca1_pca_coordinates.csv", index=False)

    if not cluster_profiles.empty:
        cluster_profiles.to_csv(output_dir / "ca1_cluster_profiles.csv", index=False)

    # Plots.
    plot_completeness_by_dataset(dataset_summary, figures_dir)
    plot_missingness_heatmap(df_ml, figures_dir)
    plot_pca(df_ml, figures_dir)
    plot_clusters(df_ml, figures_dir)

    write_interpretation_markdown(output_dir, df_ml, dataset_summary, missing_report, silhouette)

    print("\nDone.")
    print("Output directory:", output_dir)
    print("Main table:", output_dir / "ca1_harmonized_sessions.csv")
    print("Dataset summary:", output_dir / "ca1_dataset_summary.csv")
    print("Figures:", figures_dir)

    return df_ml


def main():
    parser = argparse.ArgumentParser(
        description="Run CA1 metadata ML meta-analysis from extracted metadata JSON files."
    )

    parser.add_argument(
        "json_files",
        nargs="+",
        help="Extracted metadata JSON files to combine.",
    )

    parser.add_argument(
        "--output_dir",
        default="meta_analysis/outputs",
        help="Output directory. Default: meta_analysis/outputs",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=4,
        help="Number of KMeans clusters. Default: 4",
    )

    args = parser.parse_args()

    run_pipeline(
        input_paths=args.json_files,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
    )


if __name__ == "__main__":
    main()
