# -*- coding: utf-8 -*-

"""
Create annotation pre-fill dictionaries from extracted metadata.

Aim:
    Bridge automatic extraction and the in-depth curation workflow.

This file does not create final openMINDS objects.
It creates structured dictionaries that can be manually completed
and then used by the existing in-depth curation module.
"""

import os
import json
import argparse

from utils import unique_values


def collect_subjects(metadata):
    """
    Create subject pre-fill entries.
    """

    subjects = []

    for file_entry in metadata.get("files", []):
        file_metadata = file_entry.get("file_metadata", {})
        subject_id = file_metadata.get("subject_id")

        if subject_id is not None:
            subjects.append(subject_id)

    subjects = unique_values(subjects)

    subject_entries = []

    for subject_id in subjects:
        subject_entries.append({
            "subject_id": subject_id,
            "species": None,
            "strain": None,
            "sex": None,
            "age": None,
            "biological_sex": None,
            "manual_completion_required": True,
        })

    return subject_entries


def collect_sessions(metadata):
    """
    Create session pre-fill entries.
    """

    session_entries = []

    for file_entry in metadata.get("files", []):
        file_metadata = file_entry.get("file_metadata", {})

        session_id = file_metadata.get("session_id")
        subject_id = file_metadata.get("subject_id")
        session_date = file_metadata.get("session_date")

        if session_id is not None:
            session_entries.append({
                "session_id": session_id,
                "subject_id": subject_id,
                "session_date": session_date,
            })

    unique_session_entries = []
    seen = set()

    for entry in session_entries:
        key = entry["session_id"]

        if key not in seen:
            seen.add(key)
            unique_session_entries.append(entry)

    return unique_session_entries


def collect_data_files(metadata):
    """
    Create data file entries.
    """

    data_files = []

    for file_entry in metadata.get("files", []):

        file_metadata = file_entry.get("file_metadata", {})
        neo_extraction = file_entry.get("neo_extraction") or {}
        pickle_extraction = file_entry.get("pickle_extraction") or {}
        nwb_extraction = file_entry.get("nwb_extraction") or {}

        data_files.append({
            "path": file_metadata.get("path"),
            "file_name": file_metadata.get("file_name"),
            "file_extension": file_metadata.get("file_extension"),
            "file_size_mb": file_metadata.get("file_size_mb"),
            "subject_id": file_metadata.get("subject_id"),
            "session_id": file_metadata.get("session_id"),
            "session_date": file_metadata.get("session_date"),
            "data_label": file_metadata.get("data_label"),
            "possible_signal_type": file_metadata.get("possible_signal_type"),

            "neo_attempted": neo_extraction.get("attempted"),
            "neo_success": neo_extraction.get("success"),
            "neo_error": neo_extraction.get("error"),

            "pickle_attempted": pickle_extraction.get("attempted"),
            "pickle_success": pickle_extraction.get("success"),
            "pickle_error": pickle_extraction.get("error"),

            "nwb_attempted": nwb_extraction.get("attempted"),
            "nwb_success": nwb_extraction.get("success"),
            "nwb_error": nwb_extraction.get("error"),
        })

    return data_files


def collect_recording_prefill(metadata):
    """
    Create recording dictionary based on the in-depth curation annotation style.
    """

    sampling_frequencies = []
    units = []
    channel_entries = []
    signal_types = []

    for file_entry in metadata.get("files", []):

        file_metadata = file_entry.get("file_metadata", {})
        neo_extraction = file_entry.get("neo_extraction") or {}

        signal_types.append(file_metadata.get("possible_signal_type"))

        if neo_extraction.get("success") is True:

            for sampling_rate in neo_extraction.get("sampling_rates_hz", []):
                sampling_frequencies.append(sampling_rate)

            for unit in neo_extraction.get("units", []):
                units.append(unit)

            for n_channels in neo_extraction.get("n_channels_per_segment", []):
                channel_entries.append({
                    "name": None,
                    "number_of_channels": n_channels,
                    "units": neo_extraction.get("units", []),
                })

    recording = {
        "type": "extracellular recording",
        "description": (
            "Automatically pre-filled from extracted file metadata. "
            "Manual validation and completion are required."
        ),
        "cell type": None,
        "brain region": None,
        "recording mode": None,
        "acquisition system": None,
        "channels": channel_entries,
        "sampling frequency": unique_values(sampling_frequencies),
        "units": unique_values(units),
        "detected_signal_types": unique_values(signal_types),
        "manual_completion_required": True,
    }

    return recording


def collect_experimental_context_prefill():
    """
    Manual fields that are usually not available from files.
    """

    return {
        "behavioral_paradigm": None,
        "task_conditions": None,
        "experimental_protocol": None,
        "electrode_type": None,
        "electrode_placement": None,
        "recording_depth": None,
        "anatomical_target": None,
        "surgical_procedure": None,
        "preprocessing": None,
        "manual_completion_required": True,
    }


def create_annotation_prefill(json_path, output_json=None):
    """
    Create annotation pre-fill JSON.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    prefill = {
        "dataset_identifier": {
            "dataset_name": metadata.get("dataset_name"),
            "dataset_folder": metadata.get("dataset_folder"),
            "doi": None,
            "version": None,
        },

        "dataset_summary": metadata.get("dataset_summary"),

        "subjects": collect_subjects(metadata),

        "sessions": collect_sessions(metadata),

        "data_files": collect_data_files(metadata),

        "recording": collect_recording_prefill(metadata),

        "experimental_context": collect_experimental_context_prefill(),

        "sessions_metadata": metadata.get("sessions_metadata"),

        "openminds_note": (
            "This file is an annotation pre-fill layer. "
            "It prepares metadata for the in-depth curation workflow. "
            "Manual completion is required before conversion into openMINDS objects."
        ),
    }

    if output_json is None:
        output_folder = os.path.dirname(json_path)
        output_json = os.path.join(output_folder, "annotation_prefill.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(prefill, f, indent=4)

    print("Annotation pre-fill generated")
    print("Output file:", output_json)

    return prefill


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create annotation pre-fill from extracted metadata"
    )

    parser.add_argument(
        "json_path",
        help="Path to extracted metadata JSON"
    )

    parser.add_argument(
        "--output_json",
        default=None,
        help="Path to output annotation pre-fill JSON"
    )

    args = parser.parse_args()

    create_annotation_prefill(
        json_path=args.json_path,
        output_json=args.output_json,
    )