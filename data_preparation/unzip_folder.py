# -*- coding: utf-8 -*-

"""
To facilitate data exploration and annotation.

Aim:
    Find and unzip a zip file in a chosen directory.

Usage in terminal:
    python unzip_folder.py path/to/folder

Authors:
    Talissa Kassably
Based on script by:
    Alix E. Bonard
"""

import os
import zipfile


def find_zip_file(directory):
    """
    input:
        directory: str

    output:
        zip_path: str
            path to the first zip file found
    """

    zip_files = []

    for folder_path, sub_folder, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(folder_path, file))

    if len(zip_files) == 0:
        raise FileNotFoundError("No zip file found in this directory")

    zip_path = zip_files[0]

    return zip_path


def unzip_folder(zip_path, directory=None):
    """
    input:
        zip_path: str
            path to the zip file

        directory: str
            folder where the zip file should be extracted

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