# -*- coding: utf-8 -*-

"""
Summarize extracted metadata JSON into a CSV table.

Aim:
    Convert the full extracted_metadata.json output into a readable table.

Usage:
    py extraction\\summarize_extracted_metadata.py path\\to\\extracted_metadata.json

Output:
    metadata_summary.csv
"""

import os
import json
import argparse
import pandas as pd


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

        row = {
            "dataset_name": metadata.get("dataset_name"),
            "file_name": file_metadata.get("file_name"),
            "relative_path": file_metadata.get("path"),
            "file_extension": file_metadata.get("file_extension"),
            "file_size_mb": file_metadata.get("file_size_mb"),

            "rat_id": file_metadata.get("rat_id"),
            "subject_id": file_metadata.get("subject_id") or file_metadata.get("rat_id"),
            "session_date": file_metadata.get("session_date"),
            "session_id": file_metadata.get("session_id"),
            "data_label": file_metadata.get("data_label"),
            "possible_signal_type": file_metadata.get("possible_signal_type"),

            "neo_attempted": neo_extraction.get("attempted"),
            "neo_success": neo_extraction.get("success"),
            "neo_error": neo_extraction.get("error"),
            "neo_io_class": neo_extraction.get("neo_io_class"),
            "neo_loading_mode": neo_extraction.get("loading_mode"),
            "n_segments": neo_extraction.get("n_segments"),
            "n_analogsignals": neo_extraction.get("n_analogsignals"),
            "n_spiketrains": neo_extraction.get("n_spiketrains"),
            "n_events": neo_extraction.get("n_events"),
            "n_epochs": neo_extraction.get("n_epochs"),
            "sampling_rates_hz": neo_extraction.get("sampling_rates_hz"),
            "units": neo_extraction.get("units"),
            "durations_s": neo_extraction.get("durations_s"),

            "pickle_attempted": pickle_extraction.get("attempted"),
            "pickle_success": pickle_extraction.get("success"),
            "pickle_error": pickle_extraction.get("error"),

            "nwb_attempted": nwb_extraction.get("attempted"),
            "nwb_success": nwb_extraction.get("success"),
            "nwb_error": nwb_extraction.get("error"),

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