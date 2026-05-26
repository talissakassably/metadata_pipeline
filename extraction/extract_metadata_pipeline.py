# -*- coding: utf-8 -*-

"""
Main automatic metadata extraction pipeline.

Aim:
    Extract metadata from heterogeneous neuroscience datasets.

Principle:
    - Neo is used first whenever relevant.
    - NWB-specific metadata are extracted with pynwb when the file is .nwb.
    - BIDS metadata are extracted from JSON/TSV files when --bids is used.
    - Legacy .pkl / .nio files are handled safely:
        * the pipeline attempts Neo extraction,
        * attempts pickle extraction for .pkl,
        * and always extracts file-level and filename metadata.

Usage:
    py extraction\extract_metadata_pipeline.py "path\to\dataset" --extensions .pkl .nio
    py extraction\extract_metadata_pipeline.py "path\to\dataset" --extensions .nwb
    py extraction\extract_metadata_pipeline.py "path\to\bids_dataset" --bids

Authors:
    Talissa Kassably

Based on scripts by:
    Alix E. Bonard, Andrew P. Davison
"""

import os
import re
import json
import csv
import pickle
import argparse
from datetime import datetime


# ---------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------

def make_json_safe(value):
    """
    Convert non-JSON-serializable objects into JSON-safe objects.
    """

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    return str(value)


# ---------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------

def find_files(folder, extensions):
    """
    Find files recursively in a dataset folder.

    input:
        folder: str
        extensions: list of extensions, for example [".pkl", ".nio"]

    output:
        file_list: list of file paths
    """

    file_list = []
    extensions = [ext.lower() for ext in extensions]

    for path_folder, sub_folders, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1].lower()

            if extension in extensions:
                file_list.append(os.path.join(path_folder, file))

    return sorted(file_list)


# ---------------------------------------------------------------------
# Metadata from filename and file system
# ---------------------------------------------------------------------

def parse_filename_metadata(file_path):
    """
    Extract metadata from filenames such as:
        neo_rat-14_190130.pkl
        neo_rat-14_190202_lfps.nio
        neo_rat-16_181013_lfps_clean.nio

    output:
        rat_id
        session_date
        data_label
    """

    file_name = os.path.basename(file_path)

    metadata = {
        "rat_id": None,
        "session_date": None,
        "data_label": None,
    }

    pattern = r"neo_rat-(\d+)_(\d+)(?:_(.*?))?\.[^.]+$"
    match = re.match(pattern, file_name)

    if match:
        metadata["rat_id"] = "rat-" + match.group(1)
        metadata["session_date"] = match.group(2)
        metadata["data_label"] = match.group(3)

    return metadata


def extract_file_metadata(file_path, root_dir):
    """
    Extract basic metadata from the file system.
    This always works, even if Neo cannot read the file.
    """

    stat = os.stat(file_path)

    metadata = {
        "path": os.path.relpath(file_path, root_dir),
        "file_name": os.path.basename(file_path),
        "file_extension": os.path.splitext(file_path)[1].lower(),
        "file_size_bytes": stat.st_size,
        "file_size_mb": round(stat.st_size / (1024 * 1024), 3),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata.update(parse_filename_metadata(file_path))

    return metadata


# ---------------------------------------------------------------------
# Neo extraction
# ---------------------------------------------------------------------

def read_with_neo(file_path):
    """
    Read one file with Neo.

    Some formats support lazy loading and some do not.
    Therefore, the function first tries lazy=True and then lazy=False.
    """

    from neo import get_io

    io = get_io(file_path)

    try:
        data = io.read(lazy=True)[0]
        loading_mode = "lazy"
    except Exception:
        data = io.read(lazy=False)[0]
        loading_mode = "non_lazy"

    return io, data, loading_mode


def extract_neo_metadata(file_path):
    """
    Extract technical metadata using Neo.

    output:
        dictionary containing:
            attempted
            success
            error
            Neo object structure if readable
    """

    metadata = {
        "attempted": True,
        "success": False,
        "error": None,
        "neo_io_class": None,
        "loading_mode": None,
    }

    try:
        io, data, loading_mode = read_with_neo(file_path)

        metadata["success"] = True
        metadata["neo_io_class"] = io.__class__.__name__
        metadata["loading_mode"] = loading_mode

        metadata["n_segments"] = len(data.segments)

        n_analogsignals = 0
        n_spiketrains = 0
        n_events = 0
        n_epochs = 0

        n_channels_per_segment = []
        sampling_rates = []
        units = []
        durations = []

        for segment in data.segments:

            n_analogsignals += len(getattr(segment, "analogsignals", []))
            n_spiketrains += len(getattr(segment, "spiketrains", []))
            n_events += len(getattr(segment, "events", []))
            n_epochs += len(getattr(segment, "epochs", []))

            segment_channels = 0

            for signal in getattr(segment, "analogsignals", []):

                try:
                    if len(signal.shape) == 1:
                        n_channels = 1
                    else:
                        n_channels = signal.shape[1]

                    segment_channels += n_channels
                except Exception:
                    pass

                try:
                    sampling_rates.append(signal.sampling_rate.rescale("Hz").item())
                except Exception:
                    pass

                try:
                    units.append(signal.units.dimensionality.string)
                except Exception:
                    pass

                try:
                    duration = (signal.t_stop - signal.t_start).rescale("s").item()
                    durations.append(duration)
                except Exception:
                    pass

            n_channels_per_segment.append(segment_channels)

        metadata["n_analogsignals"] = n_analogsignals
        metadata["n_spiketrains"] = n_spiketrains
        metadata["n_events"] = n_events
        metadata["n_epochs"] = n_epochs

        metadata["n_channels_per_segment"] = list(set(n_channels_per_segment))
        metadata["sampling_rates_hz"] = list(set(sampling_rates))
        metadata["units"] = list(set(units))
        metadata["durations_s"] = list(set(durations))

        metadata["has_analogsignals"] = n_analogsignals > 0
        metadata["has_spiketrains"] = n_spiketrains > 0
        metadata["has_events"] = n_events > 0
        metadata["has_epochs"] = n_epochs > 0

    except Exception as error:
        metadata["error"] = str(error)

    return make_json_safe(metadata)


# ---------------------------------------------------------------------
# Pickle extraction
# ---------------------------------------------------------------------

def summarize_python_object(obj):
    """
    Summarize a Python object loaded from pickle.

    This helps with legacy .pkl files.
    If the object is a Neo Block, it may have segments.
    """

    summary = {
        "object_type": type(obj).__name__,
        "object_module": type(obj).__module__,
    }

    if isinstance(obj, dict):
        summary["n_keys"] = len(obj)
        summary["keys"] = list(obj.keys())

    if isinstance(obj, list):
        summary["list_length"] = len(obj)

        if len(obj) > 0:
            summary["first_item_type"] = type(obj[0]).__name__
            summary["first_item_module"] = type(obj[0]).__module__

    if hasattr(obj, "segments"):

        summary["looks_like_neo_block"] = True
        summary["n_segments"] = len(obj.segments)

        n_analogsignals = 0
        n_spiketrains = 0
        n_events = 0
        n_epochs = 0

        for segment in obj.segments:
            n_analogsignals += len(getattr(segment, "analogsignals", []))
            n_spiketrains += len(getattr(segment, "spiketrains", []))
            n_events += len(getattr(segment, "events", []))
            n_epochs += len(getattr(segment, "epochs", []))

        summary["n_analogsignals"] = n_analogsignals
        summary["n_spiketrains"] = n_spiketrains
        summary["n_events"] = n_events
        summary["n_epochs"] = n_epochs

    else:
        summary["looks_like_neo_block"] = False

    return summary


def extract_pickle_metadata(file_path):
    """
    Try loading a .pkl file using pickle.

    This may fail if the pickle was created with an old Neo version.
    The error is recorded instead of stopping the pipeline.
    """

    metadata = {
        "attempted": True,
        "success": False,
        "error": None,
        "object_summary": None,
    }

    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        metadata["success"] = True
        metadata["object_summary"] = summarize_python_object(obj)

    except Exception as error:
        metadata["error"] = str(error)

    return make_json_safe(metadata)


# ---------------------------------------------------------------------
# NWB extraction with pynwb
# ---------------------------------------------------------------------

def extract_nwb_metadata(file_path):
    """
    Extract NWB-specific metadata using pynwb.
    """

    metadata = {
        "attempted": True,
        "success": False,
        "error": None,
    }

    try:
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(file_path, "r", load_namespaces=True) as io:
            nwbfile = io.read()

            metadata["success"] = True

            metadata["session_description"] = nwbfile.session_description
            metadata["identifier"] = nwbfile.identifier
            metadata["session_start_time"] = str(nwbfile.session_start_time)

            metadata["experimenter"] = list(nwbfile.experimenter) if nwbfile.experimenter else None
            metadata["institution"] = nwbfile.institution
            metadata["lab"] = nwbfile.lab

            if nwbfile.subject is not None:
                metadata["subject"] = {
                    "subject_id": nwbfile.subject.subject_id,
                    "species": nwbfile.subject.species,
                    "sex": nwbfile.subject.sex,
                    "age": nwbfile.subject.age,
                    "description": nwbfile.subject.description,
                    "strain": nwbfile.subject.strain,
                    "genotype": nwbfile.subject.genotype,
                }
            else:
                metadata["subject"] = None

            metadata["n_acquisition_objects"] = len(nwbfile.acquisition)
            metadata["acquisition_objects"] = list(nwbfile.acquisition.keys())

            metadata["n_processing_modules"] = len(nwbfile.processing)
            metadata["processing_modules"] = list(nwbfile.processing.keys())

            metadata["n_devices"] = len(nwbfile.devices)
            metadata["devices"] = list(nwbfile.devices.keys())

            metadata["n_electrode_groups"] = len(nwbfile.electrode_groups)
            metadata["electrode_groups"] = list(nwbfile.electrode_groups.keys())

            try:
                if nwbfile.electrodes is not None:
                    electrodes_df = nwbfile.electrodes.to_dataframe()
                    metadata["n_electrodes"] = len(electrodes_df)
                    metadata["electrode_columns"] = list(electrodes_df.columns)

                    if "location" in electrodes_df.columns:
                        metadata["electrode_locations"] = list(
                            set([str(x) for x in electrodes_df["location"].dropna()])
                        )
            except Exception as error:
                metadata["electrodes_error"] = str(error)

            try:
                if nwbfile.units is not None:
                    units_df = nwbfile.units.to_dataframe()
                    metadata["n_units"] = len(units_df)
                    metadata["unit_columns"] = list(units_df.columns)
            except Exception as error:
                metadata["units_error"] = str(error)

            try:
                if nwbfile.trials is not None:
                    trials_df = nwbfile.trials.to_dataframe()
                    metadata["n_trials"] = len(trials_df)
                    metadata["trial_columns"] = list(trials_df.columns)
            except Exception as error:
                metadata["trials_error"] = str(error)

    except Exception as error:
        metadata["error"] = str(error)

    return make_json_safe(metadata)


# ---------------------------------------------------------------------
# BIDS extraction
# ---------------------------------------------------------------------

def read_json_file(json_path):
    """
    Read a JSON file safely.
    """

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as error:
        return {"error": str(error)}


def read_tsv_file(tsv_path):
    """
    Read a TSV file safely.
    """

    rows = []

    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
    except Exception as error:
        rows.append({"error": str(error)})

    return rows


def extract_bids_metadata(folder):
    """
    Extract metadata from a BIDS-like folder.
    """

    metadata = {
        "attempted": True,
        "success": True,
        "bids_folder": os.path.abspath(folder),
        "dataset_description": None,
        "participants": None,
        "subject_folders": [],
        "n_subject_folders": 0,
        "json_sidecars": [],
        "n_json_sidecars": 0,
    }

    dataset_description_path = os.path.join(folder, "dataset_description.json")

    if os.path.exists(dataset_description_path):
        metadata["dataset_description"] = read_json_file(dataset_description_path)

    participants_path = os.path.join(folder, "participants.tsv")

    if os.path.exists(participants_path):
        metadata["participants"] = read_tsv_file(participants_path)

    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)

        if os.path.isdir(item_path) and item.startswith("sub-"):
            metadata["subject_folders"].append(item)

    metadata["n_subject_folders"] = len(metadata["subject_folders"])

    for path_folder, sub_folders, files in os.walk(folder):
        for file in files:
            if file.endswith(".json") and file != "dataset_description.json":

                json_path = os.path.join(path_folder, file)

                metadata["json_sidecars"].append({
                    "path": os.path.relpath(json_path, folder),
                    "content": read_json_file(json_path),
                })

    metadata["n_json_sidecars"] = len(metadata["json_sidecars"])

    return make_json_safe(metadata)


# ---------------------------------------------------------------------
# Session table extraction
# ---------------------------------------------------------------------

def find_sessions_files(folder):
    """
    Find files starting with sessions_df.
    """

    sessions_files = []

    for path_folder, sub_folders, files in os.walk(folder):
        for file in files:
            if file.startswith("sessions_df"):
                sessions_files.append(os.path.join(path_folder, file))

    return sorted(sessions_files)


def extract_sessions_metadata(folder):
    """
    Extract metadata from sessions_df files if present.
    """

    metadata = {
        "sessions_files_found": [],
        "sessions_tables": [],
    }

    sessions_files = find_sessions_files(folder)

    metadata["sessions_files_found"] = [
        os.path.relpath(path, folder)
        for path in sessions_files
    ]

    for session_file in sessions_files:

        table_summary = {
            "path": os.path.relpath(session_file, folder),
            "readable": False,
            "error": None,
            "n_rows": None,
            "n_columns": None,
            "columns": None,
            "preview": None,
        }

        try:
            import pandas as pd

            df = pd.read_csv(session_file, sep=None, engine="python")

            table_summary["readable"] = True
            table_summary["n_rows"] = len(df)
            table_summary["n_columns"] = len(df.columns)
            table_summary["columns"] = list(df.columns)
            table_summary["preview"] = df.head(5).to_dict(orient="records")

        except Exception as error:
            table_summary["error"] = str(error)

        metadata["sessions_tables"].append(table_summary)

    return make_json_safe(metadata)


# ---------------------------------------------------------------------
# File-level extraction dispatcher
# ---------------------------------------------------------------------

def extract_metadata_from_file(file_path, root_dir):
    """
    Extract metadata from one file.

    The output separates:
        - file_metadata
        - neo_extraction
        - pickle_extraction
        - nwb_extraction
        - notes
    """

    extension = os.path.splitext(file_path)[1].lower()

    metadata = {
        "file_metadata": extract_file_metadata(file_path, root_dir),
        "neo_extraction": None,
        "pickle_extraction": None,
        "nwb_extraction": None,
        "notes": [],
    }

    neo_supported_or_attempted = [
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

    if extension in neo_supported_or_attempted:
        metadata["neo_extraction"] = extract_neo_metadata(file_path)

    if extension == ".pkl":
        metadata["pickle_extraction"] = extract_pickle_metadata(file_path)

    if extension == ".nwb":
        metadata["nwb_extraction"] = extract_nwb_metadata(file_path)

    if extension == ".nio":
        metadata["notes"].append(
            ".nio is not recognized as a standard Neo IO format by the current environment. "
            "File-level metadata and Neo attempt are recorded."
        )

    return make_json_safe(metadata)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def extract_metadata_pipeline(folder, extensions=None, bids=False, output_folder=None):
    """
    Run metadata extraction on a dataset folder.
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
        "n_files_found": 0,
        "files": [],
        "sessions_metadata": None,
        "bids_metadata": None,
    }

    file_list = find_files(folder, extensions)
    dataset_metadata["n_files_found"] = len(file_list)

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

    output_json = os.path.join(
        output_folder,
        dataset_name + "_extracted_metadata.json"
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(dataset_metadata), f, indent=4)

    print("Extraction finished")
    print("Output file:", output_json)

    return dataset_metadata


# ---------------------------------------------------------------------
# Terminal execution
# ---------------------------------------------------------------------

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