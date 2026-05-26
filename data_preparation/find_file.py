# -*- coding: utf-8 -*-

"""
To facilitate data exploration and annotation.

Aim:
    Find data files in a dataset folder.

Usage in terminal:
    python find_file.py path/to/folder .nwb

Usage in notebook:
    %run path/to/find_file.py path/to/folder .nwb

Authors:
    Talissa Kassably
Based on script by:
    Alix E. Bonard
"""

import os
import json


def find_file(folder, extensions, output_json="file_list.json"):
    """
    input:
        folder: str
            path to the dataset folder

        extensions: str or list
            file extension(s), for example ".nwb" or [".nwb", ".abf"]

        output_json: str
            name of the json file where the list will be saved

    output:
        file_list: list of file paths
    """

    if isinstance(extensions, str):
        extensions = [extensions]

    file_list = []

    for path_folder, sub_folder, files in os.walk(folder):
        for file in files:
            for extension in extensions:
                if file.endswith(extension):
                    file_list.append(os.path.join(path_folder, file))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(file_list, f, indent=4)

    print("File_list generated")
    print(len(file_list), "file(s) found")

    return file_list


if __name__ == "__main__":
    import sys

    folder = sys.argv[1]
    extensions = sys.argv[2:]

    find_file(folder, extensions)
