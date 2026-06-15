# -*- coding: utf-8 -*-

"""
Extract BIDS-like neuroimaging metadata.
"""

import os
import json
import csv

from utils import make_json_safe


def read_json_file(json_path):
    """
    Read one JSON file safely.
    """

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as error:
        return {"error": str(error)}


def read_tsv_file(tsv_path):
    """
    Read one TSV file safely.
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
    input:
        folder: str

    output:
        BIDS metadata dictionary
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
        "modality_folders": [],
    }

    dataset_description_path = os.path.join(folder, "dataset_description.json")

    if os.path.exists(dataset_description_path):
        metadata["dataset_description"] = read_json_file(dataset_description_path)

    participants_path = os.path.join(folder, "participants.tsv")

    if os.path.exists(participants_path):
        metadata["participants"] = read_tsv_file(participants_path)

    modality_folders = set()

    for path_folder, sub_folders, files in os.walk(folder):

        folder_name = os.path.basename(path_folder)

        if folder_name not in ["", os.path.basename(folder)]:
            if not folder_name.startswith("sub-"):
                modality_folders.add(folder_name)

        for file in files:
            if file.endswith(".json") and file != "dataset_description.json":

                json_path = os.path.join(path_folder, file)

                metadata["json_sidecars"].append({
                    "path": os.path.relpath(json_path, folder),
                    "content": read_json_file(json_path),
                })

    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)

        if os.path.isdir(item_path) and item.startswith("sub-"):
            metadata["subject_folders"].append(item)

    metadata["n_subject_folders"] = len(metadata["subject_folders"])
    metadata["n_json_sidecars"] = len(metadata["json_sidecars"])
    metadata["modality_folders"] = sorted(list(modality_folders))

    return make_json_safe(metadata)