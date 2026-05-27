# -*- coding: utf-8 -*-

"""
Main automatic metadata extraction pipeline.

Aim:
    Extract metadata from heterogeneous neuroscience datasets.

The pipeline:
    - finds data files
    - extracts file-level metadata
    - attempts Neo extraction for electrophysiology files
    - extracts NWB-specific metadata for .nwb files
    - extracts BIDS metadata when --bids is used
    - attempts pickle extraction for .pkl files
    - extracts sessions_df metadata
    - creates dataset-level summary
    - saves full JSON output

Important:
    For NWB datasets, subject/session metadata may be inside the NWB file,
    not in the filename. Therefore, dataset_summary uses both:
        file_metadata
        nwb_extraction
"""

import os
import sys
import json
import argparse
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from data_preparation.find_file import find_file
from utils import make_json_safe, unique_values
from extract_file_metadata import extract_file_metadata
from extract_neo_metadata import extract_neo_metadata
from extract_nwb_metadata import extract_nwb_metadata
from extract_bids_metadata import extract_bids_metadata
from extract_pickle_metadata import extract_pickle_metadata
from extract_sessions_metadata import extract_sessions_metadata


def extract_metadata_from_file(file_path, root_dir):
    """
    Extract metadata from one file.

    input:
        file_path: str
        root_dir: str

    output:
        metadata: dict
    """

    file_metadata = extract_file_metadata(file_path, root_dir)
    extension = file_metadata["file_extension"]

    metadata = {
        "file_metadata": file_metadata,
        "neo_extraction": None,
        "nwb_extraction": None,
        "pickle_extraction": None,
        "notes": [],
    }

    neo_attempt_extensions = [
        ".nwb",
        ".abf",
        ".smr",
        ".plx",
        ".nev",
        ".nsx",
        ".edf",
        ".kwik",
        ".h5",
        ".mat",
        ".pkl",
        ".nio",
    ]

    if extension in neo_attempt_extensions:
        metadata["neo_extraction"] = extract_neo_metadata(file_path)

    if extension == ".nwb":
        metadata["nwb_extraction"] = extract_nwb_metadata(file_path)

    if extension == ".pkl":
        metadata["pickle_extraction"] = extract_pickle_metadata(file_path)

    if extension == ".nio":
        metadata["notes"].append(
            ".nio is not recognized as a standard Neo IO format by the current environment. "
            "File-level metadata and Neo extraction attempt are recorded."
        )

    return make_json_safe(metadata)


def get_subject_from_file_or_nwb(file_entry):
    """
    Get subject ID from filename metadata or NWB metadata.
    """

    file_metadata = file_entry.get("file_metadata", {})
    nwb_extraction = file_entry.get("nwb_extraction") or {}

    subject_id = file_metadata.get("subject_id")

    if subject_id is None and nwb_extraction.get("success") is True:
        subject = nwb_extraction.get("subject") or {}
        subject_id = subject.get("subject_id")

    return subject_id


def get_session_from_file_or_nwb(file_entry):
    """
    Get session ID from filename metadata or NWB metadata.
    """

    file_metadata = file_entry.get("file_metadata", {})
    nwb_extraction = file_entry.get("nwb_extraction") or {}

    session_id = file_metadata.get("session_id")

    if session_id is None and nwb_extraction.get("success") is True:
        session_id = nwb_extraction.get("identifier")

    return session_id


def get_session_date_from_file_or_nwb(file_entry):
    """
    Get session date from filename metadata or NWB session_start_time.
    """

    file_metadata = file_entry.get("file_metadata", {})
    nwb_extraction = file_entry.get("nwb_extraction") or {}

    session_date = file_metadata.get("session_date")

    if session_date is None and nwb_extraction.get("success") is True:
        session_start_time = nwb_extraction.get("session_start_time")

        if session_start_time is not None:
            session_date = session_start_time.split(" ")[0]

    return session_date


def create_dataset_summary(dataset_metadata):
    """
    Create dataset-level summary from extracted metadata.

    This function uses:
        - file_metadata
        - neo_extraction
        - nwb_extraction

    For NWB files, the important subject/session metadata usually comes
    from nwb_extraction rather than the filename.
    """

    subjects = []
    sessions = []
    session_dates = []
    extensions = []
    signal_types = []

    species = []
    sexes = []
    experimenters = []
    institutions = []

    session_descriptions = []
    processing_modules = []
    devices = []
    electrode_locations = []
    electrode_group_names = []
    interval_names = []

    n_electrodes_values = []
    n_units_values = []
    n_trials_values = []

    neo_success = 0
    neo_failure = 0
    pickle_success = 0
    pickle_failure = 0
    nwb_success = 0
    nwb_failure = 0

    for file_entry in dataset_metadata["files"]:

        file_metadata = file_entry.get("file_metadata", {})
        neo_extraction = file_entry.get("neo_extraction")
        pickle_extraction = file_entry.get("pickle_extraction")
        nwb_extraction = file_entry.get("nwb_extraction")

        extensions.append(file_metadata.get("file_extension"))
        signal_types.append(file_metadata.get("possible_signal_type"))

        subject_id = get_subject_from_file_or_nwb(file_entry)
        session_id = get_session_from_file_or_nwb(file_entry)
        session_date = get_session_date_from_file_or_nwb(file_entry)

        subjects.append(subject_id)
        sessions.append(session_id)
        session_dates.append(session_date)

        if neo_extraction is not None:
            if neo_extraction.get("success") is True:
                neo_success += 1
            else:
                neo_failure += 1

        if pickle_extraction is not None:
            if pickle_extraction.get("success") is True:
                pickle_success += 1
            else:
                pickle_failure += 1

        if nwb_extraction is not None:
            if nwb_extraction.get("success") is True:
                nwb_success += 1

                subject = nwb_extraction.get("subject") or {}

                species.append(subject.get("species"))
                sexes.append(subject.get("sex"))

                session_descriptions.append(nwb_extraction.get("session_description"))
                institutions.append(nwb_extraction.get("institution"))

                for experimenter in nwb_extraction.get("experimenter") or []:
                    experimenters.append(experimenter)

                for module in nwb_extraction.get("processing_modules") or []:
                    processing_modules.append(module)

                for device in nwb_extraction.get("devices") or []:
                    devices.append(device)

                for location in nwb_extraction.get("electrode_locations") or []:
                    electrode_locations.append(location)

                for group_name in nwb_extraction.get("electrode_group_names") or []:
                    electrode_group_names.append(group_name)

                for interval in nwb_extraction.get("intervals") or []:
                    interval_names.append(interval)

                n_electrodes_values.append(nwb_extraction.get("n_electrodes"))
                n_units_values.append(nwb_extraction.get("n_units"))
                n_trials_values.append(nwb_extraction.get("n_trials"))

            else:
                nwb_failure += 1

    clean_n_electrodes_values = [
        value for value in n_electrodes_values
        if isinstance(value, (int, float))
    ]

    clean_n_units_values = [
        value for value in n_units_values
        if isinstance(value, (int, float))
    ]

    clean_n_trials_values = [
        value for value in n_trials_values
        if isinstance(value, (int, float))
    ]

    summary = {
        "n_files": len(dataset_metadata["files"]),
        "file_extensions": unique_values(extensions),

        "n_subjects_detected": len(unique_values(subjects)),
        "subjects_detected": unique_values(subjects),

        "n_sessions_detected": len(unique_values(sessions)),
        "sessions_detected": unique_values(sessions),
        "session_dates_detected": unique_values(session_dates),

        "possible_signal_types": unique_values(signal_types),

        "neo_success_count": neo_success,
        "neo_failure_count": neo_failure,

        "pickle_success_count": pickle_success,
        "pickle_failure_count": pickle_failure,

        "nwb_success_count": nwb_success,
        "nwb_failure_count": nwb_failure,

        "nwb_species_detected": unique_values(species),
        "nwb_sexes_detected": unique_values(sexes),
        "nwb_experimenters_detected": unique_values(experimenters),
        "nwb_institutions_detected": unique_values(institutions),

        "nwb_processing_modules_detected": unique_values(processing_modules),
        "nwb_devices_detected": unique_values(devices),
        "nwb_electrode_locations_detected": unique_values(electrode_locations),
        "nwb_electrode_group_names_detected": unique_values(electrode_group_names),
        "nwb_intervals_detected": unique_values(interval_names),

        "nwb_session_descriptions_detected": unique_values(session_descriptions),

        "nwb_min_electrodes_per_file": min(clean_n_electrodes_values) if clean_n_electrodes_values else None,
        "nwb_max_electrodes_per_file": max(clean_n_electrodes_values) if clean_n_electrodes_values else None,

        "nwb_min_units_per_file": min(clean_n_units_values) if clean_n_units_values else None,
        "nwb_max_units_per_file": max(clean_n_units_values) if clean_n_units_values else None,
        "nwb_total_units_across_files": sum(clean_n_units_values) if clean_n_units_values else None,

        "nwb_min_trials_per_file": min(clean_n_trials_values) if clean_n_trials_values else None,
        "nwb_max_trials_per_file": max(clean_n_trials_values) if clean_n_trials_values else None,
    }

    return make_json_safe(summary)


def extract_metadata_pipeline(folder, extensions=None, bids=False, output_folder=None):
    """
    Run full automatic metadata extraction.

    input:
        folder: dataset folder
        extensions: list of file extensions
        bids: bool
        output_folder: output folder

    output:
        dataset_metadata: dict
    """

    if extensions is None or len(extensions) == 0:
        extensions = [
            ".nwb",
            ".abf",
            ".smr",
            ".plx",
            ".nev",
            ".nsx",
            ".edf",
            ".kwik",
            ".h5",
            ".mat",
            ".pkl",
            ".nio",
        ]

    extensions = [ext.lower() for ext in extensions]

    if output_folder is None:
        output_folder = os.path.join(
            os.getcwd(),
            "outputs",
            "extracted_metadata"
        )

    os.makedirs(output_folder, exist_ok=True)

    dataset_name = os.path.basename(os.path.abspath(folder))

    dataset_metadata = {
        "dataset_name": dataset_name,
        "dataset_folder": os.path.abspath(folder),
        "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extensions_searched": extensions,
        "dataset_summary": None,
        "files": [],
        "sessions_metadata": None,
        "bids_metadata": None,
    }

    file_list = find_file(folder, extensions)

    print(len(file_list), "file(s) found")

    for file_path in file_list:
        print("Extracting metadata from:", os.path.basename(file_path))
        file_metadata = extract_metadata_from_file(file_path, folder)
        dataset_metadata["files"].append(file_metadata)

    print("Extracting sessions metadata")
    dataset_metadata["sessions_metadata"] = extract_sessions_metadata(folder)

    if bids:
        print("Extracting BIDS metadata")
        dataset_metadata["bids_metadata"] = extract_bids_metadata(folder)

    dataset_metadata["dataset_summary"] = create_dataset_summary(dataset_metadata)

    output_json = os.path.join(
        output_folder,
        dataset_name + "_extracted_metadata.json"
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(dataset_metadata), f, indent=4)

    print("Extraction finished")
    print("Output file:", output_json)

    return dataset_metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Automatic metadata extraction pipeline"
    )

    parser.add_argument(
        "folder",
        help="Path to the dataset folder"
    )

    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[],
        help="File extensions to search, for example .pkl .nio .nwb"
    )

    parser.add_argument(
        "--bids",
        action="store_true",
        help="Extract BIDS metadata from the folder"
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        help="Folder where output JSON file will be saved"
    )

    args = parser.parse_args()

    extract_metadata_pipeline(
        folder=args.folder,
        extensions=args.extensions,
        bids=args.bids,
        output_folder=args.output_folder,
    )