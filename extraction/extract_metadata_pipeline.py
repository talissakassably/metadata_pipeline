# -*- coding: utf-8 -*-

"""
Main metadata extraction pipeline.

Aim:
    Run automatic metadata extraction on a dataset folder.

Workflow:
    dataset folder
        -> find data files
        -> extract metadata depending on file format
        -> save extracted metadata as JSON

Usage in terminal:
    python extract_metadata_pipeline.py path/to/dataset .nwb
    python extract_metadata_pipeline.py path/to/dataset .abf
    python extract_metadata_pipeline.py path/to/dataset .nwb .abf .smr

Usage in notebook:
    %run metadata_pipeline/extraction/extract_metadata_pipeline.py path/to/dataset .nwb

Authors:
    Talissa Kassably

Based on scripts by:
    Alix E. Bonard, Andrew P. Davison
"""

import os
import json
import sys
from datetime import datetime

from neo import get_io


# ---------------------------------------------------------------------
# 1. Find data files
# ---------------------------------------------------------------------

def find_file(folder, extensions):
    """
    Find data files in a dataset folder.

    input:
        folder: str
            path to the dataset folder

        extensions: list
            list of file extensions, for example [".nwb"] or [".abf", ".smr"]

    output:
        file_list: list
            list of file paths
    """

    file_list = []

    for path_folder, sub_folder, files in os.walk(folder):
        for file in files:
            for extension in extensions:
                if file.endswith(extension):
                    file_list.append(os.path.join(path_folder, file))

    with open("file_list.json", "w") as f:
        json.dump(file_list, f, indent=4)

    print("File_list generated")
    print(len(file_list), "file(s) found")

    return file_list


# ---------------------------------------------------------------------
# 2. Extract metadata from Neo-readable files
# ---------------------------------------------------------------------

def extract_neo_metadata(file_path, root_dir):
    """
    Extract technical metadata from one file readable by Neo.

    input:
        file_path: str
            path to one data file

        root_dir: str
            root folder of the dataset

    output:
        metadata: dict
            extracted metadata for this file
    """

    metadata = {
        "path": os.path.relpath(file_path, root_dir),
        "file_name": os.path.basename(file_path),
        "file_extension": os.path.splitext(file_path)[1],
        "readable_with_neo": False,
        "error": None,
    }

    try:
        io = get_io(file_path)
        data = io.read(lazy=True)[0]

        metadata["readable_with_neo"] = True
        metadata["neo_io_class"] = io.__class__.__name__
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

            n_analogsignals += len(segment.analogsignals)
            n_spiketrains += len(segment.spiketrains)
            n_events += len(segment.events)
            n_epochs += len(segment.epochs)

            segment_channels = 0

            for signal in segment.analogsignals:

                # signal shape is generally:
                # number of samples x number of channels
                if len(signal.shape) == 1:
                    n_channels = 1
                else:
                    n_channels = signal.shape[1]

                segment_channels += n_channels

                try:
                    sampling_rates.append(
                        signal.sampling_rate.rescale("Hz").item()
                    )
                except Exception:
                    pass

                try:
                    units.append(
                        signal.units.dimensionality.string
                    )
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

    return metadata


# ---------------------------------------------------------------------
# 3. Format-specific extraction dispatcher
# ---------------------------------------------------------------------

def extract_metadata_from_file(file_path, root_dir):
    """
    Decide which extraction method should be used depending on the file format.

    For now:
        .nwb, .abf, .smr, .plx, .nev, .nsx, .edf, etc.
        are handled through Neo when supported.

    Later:
        BIDS-specific extraction can be added here.
    """

    extension = os.path.splitext(file_path)[1]

    neo_extensions = [
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
    ]

    if extension in neo_extensions:
        metadata = extract_neo_metadata(file_path, root_dir)

    else:
        metadata = {
            "path": os.path.relpath(file_path, root_dir),
            "file_name": os.path.basename(file_path),
            "file_extension": extension,
            "readable_with_neo": False,
            "error": "No extractor implemented for this file format yet",
        }

    return metadata


# ---------------------------------------------------------------------
# 4. Run full extraction pipeline
# ---------------------------------------------------------------------

def extract_metadata_pipeline(folder, extensions):
    """
    Main metadata extraction pipeline.

    input:
        folder: str
            path to dataset folder

        extensions: list
            list of file extensions to search

    output:
        dataset_metadata: dict
            metadata extracted from all selected files
    """

    dataset_metadata = {
        "dataset_folder": os.path.abspath(folder),
        "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extensions_searched": extensions,
        "n_files_found": 0,
        "files": [],
    }

    file_list = find_file(folder, extensions)

    dataset_metadata["n_files_found"] = len(file_list)

    for file_path in file_list:
        print("Extracting metadata from:", os.path.basename(file_path))

        file_metadata = extract_metadata_from_file(file_path, folder)
        dataset_metadata["files"].append(file_metadata)

    output_name = "extracted_metadata.json"

    with open(output_name, "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    print("Extracted_metadata generated")
    print("Output file:", output_name)

    return dataset_metadata


# ---------------------------------------------------------------------
# 5. Terminal execution
# ---------------------------------------------------------------------

if __name__ == "__main__":

    folder = sys.argv[1]
    extensions = sys.argv[2:]

    if len(extensions) == 0:
        raise ValueError(
            "Please provide at least one file extension, for example: .nwb"
        )

    extract_metadata_pipeline(folder, extensions)