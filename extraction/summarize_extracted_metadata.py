# -*- coding: utf-8 -*-

"""
Summarize extracted metadata JSON into a CSV table.

Aim:
    Convert the full extracted_metadata.json output into a readable table.

This version includes:
    - file metadata
    - Neo metadata
    - NWB metadata
    - pickle metadata
    - subject/session metadata from NWB when available

Usage:
    py extraction\\summarize_extracted_metadata.py path\\to\\extracted_metadata.json

Output:
    metadata_summary.csv
"""

import os
import json
import argparse
import pandas as pd


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
            session_date = session_start_time.split(" ")[0]

    return session_date


def list_to_string(value):
    """
    Convert list values to readable CSV strings.
    """

    if value is None:
        return None

    if isinstance(value, list):
        return " | ".join([str(item) for item in value])

    return value


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
            # Subject/session metadata, combining filename + NWB
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
            # Neo extraction
            # ---------------------------------------------------------
            "neo_attempted": neo_extraction.get("attempted"),
            "neo_success": neo_extraction.get("success"),
            "neo_error": neo_extraction.get("error"),
            "neo_io_class": neo_extraction.get("neo_io_class"),
            "neo_loading_mode": neo_extraction.get("loading_mode"),
            "neo_n_segments": neo_extraction.get("n_segments"),
            "neo_n_analogsignals": neo_extraction.get("n_analogsignals"),
            "neo_n_spiketrains": neo_extraction.get("n_spiketrains"),
            "neo_n_events": neo_extraction.get("n_events"),
            "neo_n_epochs": neo_extraction.get("n_epochs"),
            "neo_sampling_rates_hz": list_to_string(neo_extraction.get("sampling_rates_hz")),
            "neo_units": list_to_string(neo_extraction.get("units")),
            "neo_durations_s": list_to_string(neo_extraction.get("durations_s")),

            # ---------------------------------------------------------
            # NWB extraction
            # ---------------------------------------------------------
            "nwb_attempted": nwb_extraction.get("attempted"),
            "nwb_success": nwb_extraction.get("success"),
            "nwb_error": nwb_extraction.get("error"),

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
            "nwb_electrode_group_names": list_to_string(nwb_extraction.get("electrode_group_names")),

            "nwb_n_units": nwb_extraction.get("n_units"),
            "nwb_unit_columns": list_to_string(nwb_extraction.get("unit_columns")),

            "nwb_n_trials": nwb_extraction.get("n_trials"),
            "nwb_trial_columns": list_to_string(nwb_extraction.get("trial_columns")),
            "nwb_intervals": list_to_string(nwb_extraction.get("intervals")),

            # ---------------------------------------------------------
            # Pickle extraction
            # ---------------------------------------------------------
            "pickle_attempted": pickle_extraction.get("attempted"),
            "pickle_success": pickle_extraction.get("success"),
            "pickle_error": pickle_extraction.get("error"),

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