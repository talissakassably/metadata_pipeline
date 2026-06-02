# -*- coding: utf-8 -*-

"""
Summarize openfield metadata JSON into one CSV.

Input:
    outputs/extracted_metadata/<dataset>_openfield_metadata.json

Output:
    outputs/extracted_metadata/<dataset>_openfield_metadata_summary.csv

This summary is one row per session.
"""

import os
import json
import argparse
import pandas as pd


def list_to_string(value):
    """
    Convert list values to a readable CSV string.
    """

    if value is None:
        return None

    if isinstance(value, list):
        return " | ".join([str(item) for item in value])

    return str(value)


def dict_to_string(value):
    """
    Convert dict values to compact CSV string.
    """

    if value is None:
        return None

    if not isinstance(value, dict):
        return str(value)

    parts = []

    for key, item in value.items():
        parts.append(f"{key}: {item}")

    return " | ".join(parts)


def make_openfield_summary(metadata):
    """
    Create one row per openfield session.
    """

    rows = []

    for session in metadata.get("sessions", []):

        file_counts = session.get("file_counts") or {}
        neuralynx = session.get("neuralynx_metadata") or {}
        axona = session.get("axona_metadata") or {}
        mclust = session.get("mclust_metadata") or {}
        optitrack = session.get("optitrack_metadata") or {}
        summary = session.get("summary") or {}

        row = {
            # Dataset/session identity
            "dataset_name": metadata.get("dataset_name"),
            "extractor": metadata.get("extractor"),
            "session_id": session.get("session_id"),
            "subject_id": session.get("subject_id"),
            "session_date": session.get("session_date"),
            "recording_system": session.get("recording_system"),
            "relative_session_folder": session.get("relative_session_folder"),

            # General recording metadata
            "recording_time": summary.get("recording_time"),
            "recording_duration_s": summary.get("recording_duration_s"),

            # Clear internal electrophysiology fields
            "raw_spike_events_total": summary.get("raw_spike_events_total"),
            "sorted_units_mclust": summary.get("sorted_units_mclust"),
            "sorted_unit_spikes_total_mclust": summary.get("sorted_unit_spikes_total_mclust"),
            "n_units_best_available": summary.get("n_units_best_available"),
            "n_units_source": summary.get("n_units_source"),
            "n_lfp_channels": summary.get("n_lfp_channels"),

            # Completeness flags
            "has_raw_spike_metadata": summary.get("has_raw_spike_metadata"),
            "has_sorted_unit_metadata": summary.get("has_sorted_unit_metadata"),
            "has_lfp_metadata": summary.get("has_lfp_metadata"),
            "has_position_metadata": summary.get("has_position_metadata"),
            "has_axona_position_metadata": summary.get("has_axona_position_metadata"),
            "has_neuralynx_position_metadata": summary.get("has_neuralynx_position_metadata"),
            "has_optitrack_metadata": summary.get("has_optitrack_metadata"),
            "has_mclust_metadata": summary.get("has_mclust_metadata"),

            # File counts
            "n_files_total": file_counts.get("n_files_total"),
            "n_ncs_files": file_counts.get("n_ncs_files"),
            "n_ntt_files": file_counts.get("n_ntt_files"),
            "n_nvt_files": file_counts.get("n_nvt_files"),
            "n_set_files": file_counts.get("n_set_files"),
            "n_pos_files": file_counts.get("n_pos_files"),
            "n_axona_spike_files": file_counts.get("n_axona_spike_files"),
            "n_eeg_files": file_counts.get("n_eeg_files"),
            "n_t64_files": file_counts.get("n_t64_files"),
            "n_fd_files": file_counts.get("n_fd_files"),
            "n_clusters_files": file_counts.get("n_clusters_files"),
            "n_csv_files": file_counts.get("n_csv_files"),
            "extensions": dict_to_string(file_counts.get("extensions")),

            # Neuralynx details
            "neuralynx_success": neuralynx.get("success"),
            "neuralynx_error": neuralynx.get("error"),
            "neuralynx_recording_name": neuralynx.get("recording_name"),
            "neuralynx_n_spike_tetrodes": neuralynx.get("n_spike_tetrodes"),
            "neuralynx_raw_spike_events_total": neuralynx.get("raw_spike_events_total"),
            "neuralynx_raw_spike_events_per_tetrode": dict_to_string(
                neuralynx.get("raw_spike_events_per_tetrode")
            ),
            "neuralynx_n_lfp_channels": neuralynx.get("n_lfp_channels"),
            "neuralynx_lfp_sampling_rates_hz": list_to_string(
                neuralynx.get("lfp_sampling_rates_hz")
            ),
            "neuralynx_position_success": neuralynx.get("position_success"),
            "neuralynx_position_error": neuralynx.get("position_error"),
            "neuralynx_events_success": neuralynx.get("events_success"),
            "neuralynx_events_error": neuralynx.get("events_error"),

            # Axona details
            "axona_success": axona.get("success"),
            "axona_error": axona.get("error"),
            "axona_recording_name": axona.get("recording_name"),
            "axona_tetrode_list": list_to_string(axona.get("tetrode_list")),
            "axona_n_spike_channels": axona.get("n_spike_channels"),
            "axona_raw_spike_events_total": axona.get("raw_spike_events_total"),
            "axona_raw_spike_events_per_channel": dict_to_string(
                axona.get("raw_spike_events_per_channel")
            ),
            "axona_position_success": axona.get("position_success"),
            "axona_position_sample_rate_hz": axona.get("position_sample_rate_hz"),
            "axona_n_position_samples": axona.get("n_position_samples"),
            "axona_position_coordinates_shape": list_to_string(
                axona.get("position_coordinates_shape")
            ),
            "axona_n_eeg_channels": axona.get("n_eeg_channels"),
            "axona_eeg_sample_rates_hz": list_to_string(
                axona.get("eeg_sample_rates_hz")
            ),
            "axona_eeg_signal_shapes": dict_to_string(
                axona.get("eeg_signal_shapes")
            ),

            # MClust details
            "mclust_success": mclust.get("success"),
            "mclust_error": mclust.get("error"),
            "mclust_n_tetrodes_with_units": mclust.get("n_tetrodes_with_units"),
            "mclust_sorted_units": mclust.get("sorted_units_mclust"),
            "mclust_sorted_unit_spikes_total": mclust.get("sorted_unit_spikes_total_mclust"),
            "mclust_units_per_tetrode": dict_to_string(
                mclust.get("units_per_tetrode")
            ),
            "mclust_unit_header_keys": list_to_string(
                mclust.get("unit_header_keys")
            ),

            # OptiTrack details
            "optitrack_success": optitrack.get("success"),
            "optitrack_error": optitrack.get("error"),
            "optitrack_n_csv_files": optitrack.get("n_csv_files"),
            "optitrack_csv_files": list_to_string(optitrack.get("csv_files")),
            "optitrack_tracking_name": optitrack.get("tracking_name"),
            "optitrack_recording_time": optitrack.get("recording_time"),
            "optitrack_sample_rate_hz": optitrack.get("sample_rate_hz"),
            "optitrack_rigid_body_names": list_to_string(
                optitrack.get("rigid_body_names")
            ),
            "optitrack_n_timestamps": optitrack.get("n_timestamps"),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_openfield_metadata(json_path, output_csv=None):
    """
    Generate openfield summary CSV.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df = make_openfield_summary(metadata)

    if output_csv is None:
        output_csv = json_path.replace(".json", "_summary.csv")

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print("Openfield summary generated")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Output:", output_csv)

    preview_columns = [
        "session_id",
        "subject_id",
        "recording_system",
        "recording_duration_s",
        "raw_spike_events_total",
        "sorted_units_mclust",
        "sorted_unit_spikes_total_mclust",
        "n_lfp_channels",
        "has_position_metadata",
        "has_optitrack_metadata",
    ]

    available_preview_columns = [
        column for column in preview_columns
        if column in df.columns
    ]

    print("\nPreview:")
    print(df[available_preview_columns].head())

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Summarize openfield metadata JSON into one CSV"
    )

    parser.add_argument(
        "json_path",
        help="Path to openfield metadata JSON"
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional output CSV path"
    )

    args = parser.parse_args()

    summarize_openfield_metadata(
        json_path=args.json_path,
        output_csv=args.output_csv,
    )