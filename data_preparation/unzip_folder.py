# -*- coding: utf-8 -*-

"""
Unzip a downloaded dataset folder.

Aim:
    Find and unzip a zip file in a chosen directory.
"""

import os
import zipfile


def find_zip_file(directory):
    """
    input:
        directory: str

    output:
        zip_path: str
    """

    zip_files = []

    for folder_path, sub_folders, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(folder_path, file))

    if len(zip_files) == 0:
        raise FileNotFoundError("No zip file found in this directory")

    return zip_files[0]


def unzip_folder(zip_path, directory=None):
    """
    input:
        zip_path: str
        directory: str or None

    output:
        extracted_folder: str
    """

    if directory is None:
        directory = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(directory)

        extracted_files = zip_ref.namelist()
        top_level_dirs = [
            os.path.join(directory, f.split("/")[0])
            for f in extracted_files
            if len(f.split("/")[0]) > 0
        ]

        top_level_dirs = list(set(top_level_dirs))

    if len(top_level_dirs) > 0:
        return top_level_dirs[0]

    return directory


if __name__ == "__main__":
    import sys

    directory = sys.argv[1]
    zip_path = find_zip_file(directory)
    extracted_folder = unzip_folder(zip_path, directory)

    print("Zip file extracted")
    print("Extracted folder:", extracted_folder)