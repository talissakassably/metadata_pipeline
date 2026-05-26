# -*- coding: utf-8 -*-

"""
Find data files in a dataset folder.

Aim:
    Recursively find files with selected extensions.

Usage:
    py data_preparation/find_file.py path/to/folder .nwb .pkl
"""

import os
import json


def find_file(folder, extensions, output_json=None):
    """
    input:
        folder: str
        extensions: str or list of str
        output_json: optional path to save file list

    output:
        file_list: list of file paths
    """

    if isinstance(extensions, str):
        extensions = [extensions]

    extensions = [ext.lower() for ext in extensions]

    file_list = []

    for path_folder, sub_folders, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1].lower()

            if extension in extensions:
                file_list.append(os.path.join(path_folder, file))

    file_list = sorted(file_list)

    if output_json is not None:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(file_list, f, indent=4)

    print("File list generated")
    print(len(file_list), "file(s) found")

    return file_list


if __name__ == "__main__":
    import sys

    folder = sys.argv[1]
    extensions = sys.argv[2:]

    find_file(folder, extensions, output_json="file_list.json")