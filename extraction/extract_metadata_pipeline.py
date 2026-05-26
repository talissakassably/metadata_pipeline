# -*- coding: utf-8 -*-

"""
Main metadata extraction pipeline.

Aim:
    Run automatic metadata extraction on a dataset folder.

Workflow:
    dataset folder
        -> find data files
        -> extract metadata depending on file format
        -> optionally extract BIDS metadata
        -> save extracted metadata as JSON

Usage in terminal:
    py extraction/extract_metadata_pipeline.py path/to/dataset --extensions .nwb
    py extraction/extract_metadata_pipeline.py path/to/dataset --extensions .nio .pkl
    py extraction/extract_metadata_pipeline.py path/to/bids_dataset --bids

Usage in notebook:
    %run metadata_pipeline/extraction/extract_metadata_pipeline.py path/to/dataset --extensions .nwb

Authors:
    Talissa Kassably

Based on scripts by:
    Alix E. Bonard, Andrew P. Davison
"""

import os
import sys
import json
import argparse
from datetime import datetime


# ---------------------------------------------------------------------
# Import local modules
# ---------------------------------------------------------------------

# Current file:
# metadata_pipeline/extraction/extract_metadata_pipeline.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parent folder:
# metadata_pipeline/
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)

# Add folders to Python path so local imports work when running the script
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data_preparation.find_file import find_file
from extract_neo_metadata import extract_neo_metadata
from extract_bids_metadata import extract_bids_metadata

try:
    from extract_nwb_metadata import extract_nwb_metadata
except ImportError:
    extract_nwb_metadata = None


# ---------------------------------------------------------------------
# Format-specific extraction dispatcher
# ---------------------------------------------------------------------

def extract_metadata_from_file(file_path, root_dir):
    """
    Decide which extraction method should be used depending on the file format.

    For now:
        Most electrophysiology formats are handled through Neo when supported.

    NWB:
        If the file is .nwb, the pipeline tries to extract both:
            - general Neo metadata
            - NWB-specific metadata using pynwb

    Later:
        More specific extractors can be added here for .pkl, .nio, etc.
    """

    extension = os.path.splitext(file_path)[1].lower()

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
        ".nio",
        ".pkl",
    ]

    if extension == ".nwb":

        metadata = {
            "path": os.path.relpath(file_path, root_dir),
            "file_name": os.path.basename(file_path),
            "file_extension": extension,
            "neo_metadata": extract_neo_metadata(file_path, root_dir),
            "nwb_metadata": None,
        }

        if extract_nwb_metadata is not None:
            metadata["nwb_metadata"] = extract_nwb_metadata(file_path, root_dir)
        else:
            metadata["nwb_metadata"] = {
                "readable_with_pynwb": False,
                "error": "extract_nwb_metadata.py not found or pynwb import failed",
            }

    elif extension in neo_extensions:

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
# Main extraction pipeline
# ---------------------------------------------------------------------

def extract_metadata_pipeline(folder, extensions=None, bids=False, output_folder=None):
    """
    input:
        folder: str
            path to the dataset folder

        extensions: list
            list of file extensions to search for electrophysiology data

        bids: bool
            if True, extract BIDS-like metadata from the folder

        output_folder: str
            folder where outputs will be saved

    output:
        dataset_metadata: dict
    """

    if extensions is None:
        extensions = []

    if output_folder is None:
        output_folder = os.path.join(
            PIPELINE_DIR,
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
        "bids_extraction": bids,
        "n_files_found": 0,
        "files": [],
        "bids_metadata": None,
    }

    # ---------------------------------------------------------
    # Extract metadata from electrophysiology files
    # ---------------------------------------------------------

    if len(extensions) > 0:

        file_list = find_file(
            folder=folder,
            extensions=extensions,
            output_json=os.path.join(output_folder, dataset_name + "_file_list.json")
        )

        dataset_metadata["n_files_found"] = len(file_list)

        for file_path in file_list:
            print("Extracting metadata from:", os.path.basename(file_path))

            file_metadata = extract_metadata_from_file(file_path, folder)
            dataset_metadata["files"].append(file_metadata)

    # ---------------------------------------------------------
    # Extract metadata from BIDS-like dataset
    # ---------------------------------------------------------

    if bids:
        print("Extracting BIDS metadata from:", folder)
        bids_metadata = extract_bids_metadata(folder)
        dataset_metadata["bids_metadata"] = bids_metadata

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------

    output_json = os.path.join(output_folder, dataset_name + "_extracted_metadata.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=4)

    print("Extracted metadata generated")
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
        help="File extensions to search, for example .nwb .abf .smr .nio .pkl"
    )

    parser.add_argument(
        "--bids",
        action="store_true",
        help="Extract BIDS metadata from the folder"
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        help="Folder where output JSON files will be saved"
    )

    args = parser.parse_args()

    extract_metadata_pipeline(
        folder=args.folder,
        extensions=args.extensions,
        bids=args.bids,
        output_folder=args.output_folder
    )