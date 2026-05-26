# -*- coding: utf-8 -*-

"""
Extract file-level metadata.

This always works, even when Neo/NWB/BIDS extraction fails.
"""

import os
import re
from datetime import datetime


def parse_filename_metadata(file_path):
    """
    Extract subject/session information from filenames such as:
        neo_rat-14_190130.pkl
        neo_rat-14_190202_lfps.nio
        neo_rat-16_181013_lfps_clean.nio
    """

    file_name = os.path.basename(file_path)

    metadata = {
        "subject_id": None,
        "rat_id": None,
        "session_date": None,
        "session_id": None,
        "data_label": None,
        "possible_signal_type": None,
    }

    pattern = r"neo_rat-(\d+)_(\d+)(?:_(.*?))?\.[^.]+$"
    match = re.match(pattern, file_name)

    if match:
        rat_id = "rat-" + match.group(1)
        session_date = match.group(2)
        data_label = match.group(3)

        metadata["subject_id"] = rat_id
        metadata["rat_id"] = rat_id
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
            metadata["possible_signal_type"] = "single-unit or session data"

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