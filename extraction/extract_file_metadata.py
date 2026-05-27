# -*- coding: utf-8 -*-

"""
Extract file-level metadata.

This extractor always works, even when Neo/NWB/pickle extraction fails.

Supported filename/path patterns:
    neo_rat-14_190130.pkl
    neo_rat-14_190202_lfps.nio

    hbp-01681_TouchAndSee_Ramachandran_samp20__2012-08-09.pkl
    TouchAndSee/Ramachandran/samp20/hbp-01681_TouchAndSee_Ramachandran_samp20__2012-08-09.pkl
"""

import os
import re
from datetime import datetime


def parse_filename_metadata(file_path):
    """
    Extract metadata from known neuroscience dataset filename patterns.
    """

    file_name = os.path.basename(file_path)
    normalized_path = file_path.replace("\\", "/")

    metadata = {
        "subject_id": None,
        "rat_id": None,
        "animal_id": None,
        "animal_name": None,
        "session_date": None,
        "session_id": None,
        "sample_id": None,
        "data_label": None,
        "possible_signal_type": None,
        "dataset_code": None,
        "project_label": None,
        "task_label": None,
    }

    # ---------------------------------------------------------
    # Pattern 1:
    # neo_rat-14_190130.pkl
    # neo_rat-14_190202_lfps.nio
    # ---------------------------------------------------------

    rat_pattern = r"^neo_rat-(\d+)_(\d+)(?:_(.*?))?\.[^.]+$"
    rat_match = re.match(rat_pattern, file_name)

    if rat_match is not None:
        rat_id = "rat-" + rat_match.group(1)
        session_date = rat_match.group(2)
        data_label = rat_match.group(3)

        metadata["subject_id"] = rat_id
        metadata["rat_id"] = rat_id
        metadata["animal_id"] = rat_id
        metadata["animal_name"] = rat_id
        metadata["session_date"] = session_date
        metadata["session_id"] = rat_id + "_" + session_date
        metadata["data_label"] = data_label

        if data_label is not None:
            if "lfp" in data_label.lower():
                metadata["possible_signal_type"] = "local field potential"
            elif "spike" in data_label.lower():
                metadata["possible_signal_type"] = "spike data"
            else:
                metadata["possible_signal_type"] = data_label
        else:
            metadata["possible_signal_type"] = "session-level electrophysiology object"

        return metadata

    # ---------------------------------------------------------
    # Pattern 2:
    # hbp-01681_TouchAndSee_Ramachandran_samp20__2012-08-09.pkl
    # ---------------------------------------------------------

    hbp_pattern = (
        r"^(hbp-\d+)_"
        r"([^_]+)_"
        r"([^_]+)_"
        r"(samp\d+)__"
        r"(\d{4}-\d{2}-\d{2})"
        r"\.[^.]+$"
    )

    hbp_match = re.match(hbp_pattern, file_name)

    if hbp_match is not None:
        dataset_code = hbp_match.group(1)
        project_label = hbp_match.group(2)
        animal_name = hbp_match.group(3)
        sample_id = hbp_match.group(4)
        session_date = hbp_match.group(5)

        session_id = animal_name + "_" + sample_id + "_" + session_date

        metadata["dataset_code"] = dataset_code
        metadata["project_label"] = project_label
        metadata["task_label"] = project_label
        metadata["animal_name"] = animal_name
        metadata["animal_id"] = animal_name
        metadata["subject_id"] = animal_name
        metadata["sample_id"] = sample_id
        metadata["session_date"] = session_date
        metadata["session_id"] = session_id
        metadata["data_label"] = project_label
        metadata["possible_signal_type"] = "electrophysiology session object"

        return metadata

    # ---------------------------------------------------------
    # Pattern 3 fallback from folder structure:
    # TouchAndSee/Ramachandran/samp20/...
    # ---------------------------------------------------------

    path_pattern = r"TouchAndSee/([^/]+)/(samp\d+)/.*__(\d{4}-\d{2}-\d{2})\.[^.]+$"
    path_match = re.search(path_pattern, normalized_path)

    if path_match is not None:
        animal_name = path_match.group(1)
        sample_id = path_match.group(2)
        session_date = path_match.group(3)

        session_id = animal_name + "_" + sample_id + "_" + session_date

        metadata["dataset_code"] = "hbp-01681"
        metadata["project_label"] = "TouchAndSee"
        metadata["task_label"] = "TouchAndSee"
        metadata["animal_name"] = animal_name
        metadata["animal_id"] = animal_name
        metadata["subject_id"] = animal_name
        metadata["sample_id"] = sample_id
        metadata["session_date"] = session_date
        metadata["session_id"] = session_id
        metadata["data_label"] = "TouchAndSee"
        metadata["possible_signal_type"] = "electrophysiology session object"

        return metadata

    return metadata


def extract_file_metadata(file_path, root_dir):
    """
    input:
        file_path: str
        root_dir: str

    output:
        metadata: dict
    """

    stat = os.stat(file_path)
    extension = os.path.splitext(file_path)[1].lower()

    metadata = {
        "path": os.path.relpath(file_path, root_dir),
        "absolute_path": os.path.abspath(file_path),
        "file_name": os.path.basename(file_path),
        "file_extension": extension,
        "file_format_label": extension.replace(".", ""),
        "file_size_bytes": stat.st_size,
        "file_size_mb": round(stat.st_size / (1024 * 1024), 3),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata.update(parse_filename_metadata(file_path))

    return metadata