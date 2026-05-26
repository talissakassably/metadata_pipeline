# -*- coding: utf-8 -*-

"""
To facilitate automatic metadata extraction from BIDS-like neuroimaging datasets.

Aim:
    Extract simple metadata from BIDS folders:
        - dataset_description.json
        - JSON sidecar files
        - participants.tsv if present

This module is used by:
    extract_metadata_pipeline.py

Authors:
    Talissa Kassably
"""

import os
import json
import csv


def read_json_file(json_path):
    """
    input:
        json_path: str

    output:
        content: dict
    """

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
    except Exception as error:
        content = {
            "error": str(error)
        }

    return content


def read_tsv_file(tsv_path):
    """
    input:
        tsv_path: str

    output:
        rows: list of dictionaries
    """

    rows = []

    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
    except Exception as error:
        rows.append({
            "error": str(error)
        })

    return rows


def extract_bids_metadata(folder):
    """
    input:
        folder: str
            path to a BIDS-like dataset folder

    output:
        metadata: dict
            extracted BIDS metadata
    """

    metadata = {
        "bids_folder": os.path.abspath(folder),
        "dataset_description": None,
        "participants": None,
        "json_sidecars": [],
        "n_json_sidecars": 0,
        "n_subject_folders": 0,
        "subject_folders": [],
    }

    dataset_description_path = os.path.join(folder, "dataset_description.json")

    if os.path.exists(dataset_description_path):
        metadata["dataset_description"] = read_json_file(dataset_description_path)

    participants_path = os.path.join(folder, "participants.tsv")

    if os.path.exists(participants_path):
        metadata["participants"] = read_tsv_file(participants_path)

    subject_folders = []

    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path) and item.startswith("sub-"):
            subject_folders.append(item)

    metadata["subject_folders"] = subject_folders
    metadata["n_subject_folders"] = len(subject_folders)

    json_sidecars = []

    for path_folder, sub_folder, files in os.walk(folder):
        for file in files:
            if file.endswith(".json") and file != "dataset_description.json":
                json_path = os.path.join(path_folder, file)
                json_sidecars.append({
                    "path": os.path.relpath(json_path, folder),
                    "content": read_json_file(json_path)
                })

    metadata["json_sidecars"] = json_sidecars
    metadata["n_json_sidecars"] = len(json_sidecars)

    return metadata