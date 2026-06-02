# -*- coding: utf-8 -*-

"""
Summarize extracted metadata JSON into one CSV table.

This is for the regular extraction pipeline.

The output contains:
    - file metadata
    - subject/session metadata
    - Neo internal electrophysiology metadata
    - NWB internal electrophysiology metadata
    - readability/error status
"""

import os
import json
import argparse
import pandas as pd


def list_to_string(value):
    """
    Convert list values to readable CSV strings.
    """

    if value is None:
        return None

    if isinstance(value, list):
        return " | ".join([str(item) for item in value])

    return value


def dict_to_string(value):
    """
    Convert dict values to readable compact strings.
    """

    if value is None:
        return None

    if not isinstance(value, dict):
        return str(value)

    parts = []

    for key, item in value.items():
        parts.append(f"{key}: {item}")

    return " | ".join(parts)


def get_subject_from_file_or_nwb(file_metadata, nwb_extraction):
    """
    Get subject ID from file metadata or NWB metadata.
    """

    subject_id = file_metadata.get("subject_id")

    if subject_id is None and nwb_extraction:
        subject = nwb_extraction.get("subject") or {}
        subject_id = subject.get("subject_id")

    return subject_id


def get_session_from_file_or_nwb(file_metadata, nwb_extraction):
    """
    Get session ID from file metadata or NWB metadata.
    """

    session_id = file_metadata.get("session_id")

    if session_id is None and nwb_extraction:
        session_id = nwb_extraction.get("identifier")

    return session_id


def get_session_date_from_file_or_nwb(file_metadata, nwb_extraction):
    """
    Get session date from file metadata or NWB session_start_time.
    """

    session_date = file_metadata.get("session_date")

    if session_date is None and nwb_extraction:
        session_start_time = nwb_extraction.get("session_start_time")

        if session_start_time is not None:
            session_date = str(session_start_time).split(" ")[0]

    return session_date


def choose_internal_value(neo_value, nwb_value):
    """
    Prefer NWB value when available, otherwise Neo value.
    """

    if nwb_value is not None:
        return nwb_value

    return neo_value


def summarize_metadata(json_path, output_csv=None):
    """
    input:
        json_path: str
            path to extracted metadata JSON

        output_csv: str or None
            path to output CSV file

    output:
        dataframe containing one row per file
    """

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    rows = []

    for file_entry in metadata.get("files", []):

        file_metadata = file_entry.get("file_metadata", {})
        neo_extraction = file_entry.get("neo_extraction") or {}
        pickle_extraction = file_entry.get("pickle_extraction") or {}
        nwb_extraction = file_entry.get("nwb_extraction") or {}

        nwb_subject = nwb_extraction.get("subject") or {}

        row = {
            # ---------------------------------------------------------
            # Dataset-level information
            # ---------------------------------------------------------
            "dataset_name": metadata.get("dataset_name"),
            "dataset_folder": metadata.get("dataset_folder"),

            # ---------------------------------------------------------
            # File-level metadata
            # ---------------------------------------------------------
            "file_name": file_metadata.get("file_name"),
            "relative_path": file_metadata.get("path"),
            "file_extension": file_metadata.get("file_extension"),
            "file_format_label": file_metadata.get("file_format_label"),
            "file_size_mb": file_metadata.get("file_size_mb"),
            "last_modified": file_metadata.get("last_modified"),

            # ---------------------------------------------------------
            # Subject/session metadata
            # ---------------------------------------------------------
            "subject_id": get_subject_from_file_or_nwb(file_metadata, nwb_extraction),
            "session_id": get_session_from_file_or_nwb(file_metadata, nwb_extraction),
            "session_date": get_session_date_from_file_or_nwb(file_metadata, nwb_extraction),

            "rat_id": file_metadata.get("rat_id"),
            "animal_id": file_metadata.get("animal_id"),
            "animal_name": file_metadata.get("animal_name"),
            "sample_id": file_metadata.get("sample_id"),
            "dataset_code": file_metadata.get("dataset_code"),
            "project_label": file_metadata.get("project_label"),
            "task_label": file_metadata.get("task_label"),
            "data_label": file_metadata.get("data_label"),
            "possible_signal_type": file_metadata.get("possible_signal_type"),

            # ---------------------------------------------------------
            # Extraction status
            # ---------------------------------------------------------
            "neo_attempted": neo_extraction.get("attempted"),
            "neo_success": neo_extraction.get("success"),
            "neo_error": neo_extraction.get("error"),
            "neo_io_class": neo_extraction.get("neo_io_class"),
            "neo_loading_mode": neo_extraction.get("loading_mode"),

            "nwb_attempted": nwb_extraction.get("attempted"),
            "nwb_success": nwb_extraction.get("success"),
            "nwb_error": nwb_extraction.get("error"),

            "pickle_attempted": pickle_extraction.get("attempted"),
            "pickle_success": pickle_extraction.get("success"),
            "pickle_error": pickle_extraction.get("error"),

            # ---------------------------------------------------------
            # Unified internal electrophysiology metadata
            # ---------------------------------------------------------
            "n_segments": neo_extraction.get("n_segments"),

            "n_analogsignals": neo_extraction.get("n_analogsignals"),
            "n_spiketrains": neo_extraction.get("n_spiketrains"),
            "n_events": neo_extraction.get("n_events"),
            "n_epochs": neo_extraction.get("n_epochs"),

            "n_units": choose_internal_value(
                neo_extraction.get("n_spiketrains"),
                nwb_extraction.get("n_units"),
            ),

            "n_spikes_total": choose_internal_value(
                neo_extraction.get("n_spikes_total"),
                nwb_extraction.get("n_spikes_total"),
            ),

            "min_spikes_per_unit": choose_internal_value(
                neo_extraction.get("min_spikes_per_spiketrain"),
                nwb_extraction.get("min_spikes_per_unit"),
            ),

            "max_spikes_per_unit": choose_internal_value(
                neo_extraction.get("max_spikes_per_spiketrain"),
                nwb_extraction.get("max_spikes_per_unit"),
            ),

            "mean_spikes_per_unit": choose_internal_value(
                neo_extraction.get("mean_spikes_per_spiketrain"),
                nwb_extraction.get("mean_spikes_per_unit"),
            ),

            "event_names": list_to_string(neo_extraction.get("event_names")),
            "epoch_names": list_to_string(neo_extraction.get("epoch_names")),

            "sampling_rates_hz": list_to_string(neo_extraction.get("sampling_rates_hz")),
            "signal_units": list_to_string(neo_extraction.get("units")),
            "signal_durations_s": list_to_string(neo_extraction.get("durations_s")),
            "signal_names": list_to_string(neo_extraction.get("signal_names")),
            "signal_shapes": list_to_string(neo_extraction.get("signal_shapes")),
            "n_channels_per_segment": list_to_string(neo_extraction.get("n_channels_per_segment")),

            # ---------------------------------------------------------
            # NWB metadata
            # ---------------------------------------------------------
            "nwb_identifier": nwb_extraction.get("identifier"),
            "nwb_session_description": nwb_extraction.get("session_description"),
            "nwb_session_start_time": nwb_extraction.get("session_start_time"),

            "nwb_experimenter": list_to_string(nwb_extraction.get("experimenter")),
            "nwb_institution": nwb_extraction.get("institution"),
            "nwb_lab": nwb_extraction.get("lab"),
            "nwb_related_publications": list_to_string(nwb_extraction.get("related_publications")),

            "nwb_subject_id": nwb_subject.get("subject_id"),
            "nwb_subject_species": nwb_subject.get("species"),
            "nwb_subject_sex": nwb_subject.get("sex"),
            "nwb_subject_age": nwb_subject.get("age"),
            "nwb_subject_description": nwb_subject.get("description"),
            "nwb_subject_strain": nwb_subject.get("strain"),
            "nwb_subject_genotype": nwb_subject.get("genotype"),

            "nwb_n_acquisition_objects": nwb_extraction.get("n_acquisition_objects"),
            "nwb_acquisition_objects": list_to_string(nwb_extraction.get("acquisition_objects")),

            "nwb_n_processing_modules": nwb_extraction.get("n_processing_modules"),
            "nwb_processing_modules": list_to_string(nwb_extraction.get("processing_modules")),

            "nwb_n_devices": nwb_extraction.get("n_devices"),
            "nwb_devices": list_to_string(nwb_extraction.get("devices")),

            "nwb_n_electrode_groups": nwb_extraction.get("n_electrode_groups"),
            "nwb_electrode_groups": list_to_string(nwb_extraction.get("electrode_groups")),

            "nwb_n_electrodes": nwb_extraction.get("n_electrodes"),
            "nwb_electrode_columns": list_to_string(nwb_extraction.get("electrode_columns")),
            "nwb_electrode_locations": list_to_string(nwb_extraction.get("electrode_locations")),
            "nwb_electrode_location_counts": dict_to_string(nwb_extraction.get("electrode_location_counts")),
            "nwb_electrode_group_names": list_to_string(nwb_extraction.get("electrode_group_names")),
            "nwb_electrode_group_counts": dict_to_string(nwb_extraction.get("electrode_group_counts")),

            "nwb_unit_columns": list_to_string(nwb_extraction.get("unit_columns")),
            "nwb_unit_quality_columns": list_to_string(nwb_extraction.get("unit_quality_columns")),
            "nwb_unit_sampling_rates": list_to_string(nwb_extraction.get("unit_sampling_rates")),
            "nwb_cluster_quality_values": list_to_string(nwb_extraction.get("cluster_quality_values")),
            "nwb_cluster_quality_counts": dict_to_string(nwb_extraction.get("cluster_quality_counts")),
            "nwb_numeric_unit_summaries": dict_to_string(nwb_extraction.get("numeric_unit_summaries")),

            "nwb_n_trials": nwb_extraction.get("n_trials"),
            "nwb_trial_columns": list_to_string(nwb_extraction.get("trial_columns")),
            "nwb_trial_value_summaries": dict_to_string(nwb_extraction.get("trial_value_summaries")),

            "nwb_intervals": list_to_string(nwb_extraction.get("intervals")),
            "nwb_interval_summaries": list_to_string(nwb_extraction.get("interval_summaries")),

            # ---------------------------------------------------------
            # Completeness flags
            # ---------------------------------------------------------
            "has_subject_metadata": nwb_extraction.get("has_subject_metadata"),
            "has_electrode_metadata": nwb_extraction.get("has_electrode_metadata"),
            "has_unit_metadata": nwb_extraction.get("has_unit_metadata"),
            "has_spike_metadata": choose_internal_value(
                neo_extraction.get("has_spike_metadata"),
                nwb_extraction.get("has_spike_metadata"),
            ),
            "has_trial_metadata": nwb_extraction.get("has_trial_metadata"),
            "has_interval_metadata": nwb_extraction.get("has_interval_metadata"),

            # ---------------------------------------------------------
            # Notes
            # ---------------------------------------------------------
            "notes": " | ".join(file_entry.get("notes", [])),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_csv is None:
        output_folder = os.path.dirname(json_path)
        output_csv = os.path.join(output_folder, "metadata_summary.csv")

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print("Metadata summary generated")
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))
    print("Output file:", output_csv)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Summarize extracted metadata JSON into CSV"
    )

    parser.add_argument(
        "json_path",
        help="Path to extracted metadata JSON"
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    summarize_metadata(
        json_path=args.json_path,
        output_csv=args.output_csv,
    )