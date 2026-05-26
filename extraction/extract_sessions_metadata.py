# -*- coding: utf-8 -*-

"""
Extract session-level metadata from sessions_df files.
"""

import os

from utils import make_json_safe


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
    input:
        folder: str

    output:
        session metadata dictionary
    """

    metadata = {
        "sessions_files_found": [],
        "sessions_tables": [],
        "subjects_from_sessions": [],
        "session_ids_from_sessions": [],
        "n_sessions_from_tables": 0,
    }

    sessions_files = find_sessions_files(folder)

    metadata["sessions_files_found"] = [
        os.path.relpath(path, folder)
        for path in sessions_files
    ]

    all_subjects = []
    all_sessions = []

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

            metadata["n_sessions_from_tables"] += len(df)

            if "animal" in df.columns:
                all_subjects.extend(list(df["animal"].dropna().astype(str)))

            if "date" in df.columns and "animal" in df.columns:
                for _, row in df.iterrows():
                    all_sessions.append(str(row["animal"]) + "_" + str(row["date"]))

        except Exception as error:
            table_summary["error"] = str(error)

        metadata["sessions_tables"].append(table_summary)

    metadata["subjects_from_sessions"] = sorted(list(set(all_subjects)))
    metadata["session_ids_from_sessions"] = sorted(list(set(all_sessions)))

    return make_json_safe(metadata)