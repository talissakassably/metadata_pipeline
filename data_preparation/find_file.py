# -*- coding: utf-8 -*-

"""
To facilitate data exploration and anotation.

Aim: find data files in unzip folders. 
Usage in jupyter notebooks: %run path/to/find_file.py folder extension 

Authors: Alix E. Bonard
Date: 03/10/2024
Last update: 03/10/2024
"""
import os
import json 

def find_file(folder,extension):
    """
    input: folder: str 
            extension: str: .pxp or .nwb or .abf or see https://neo.readthedocs.io/ :  List of implemented IO modules
    output: list of path file 
    
    """

    file_list = []    
    for path_folder, sub_folder, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(path_folder, file))
    with open('file_list.json', "w") as f:
        json.dump(file_list, f)
    print('File_list generated')   
    
    return file_list

if __name__ == "__main__":
    import sys

    folder = sys.argv[1]
    extension = sys.argv[2]

    find_file(folder, extension)
