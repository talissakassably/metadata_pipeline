# -*- coding: utf-8 -*-

""" 
To facilitate data exploration and anotation.

Aim: Find the zip file in the chosen directory (defined here as the working directory) and unzip the zipfile. 
Usage in jupyter notebooks: %run path/to/unzip_file.py 

Authors: Alix E. Bonard
Date: 21/10/2024
Last update: 21/10/2024

"""

import os 
import glob 
import zipfile


def find_zip_file(directory): # Find zip file in the folder
    zipfile = []
    for folder_path, sub_folder, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                zipfile.append(os.path.join(folder_path, file))
    zip_path = zipfile[0]
    return zip_path   

def unzip_folder(zip_path, directory): # Unzip and extract the zipfile in the chosen directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(directory)
        extracted_files = zip_ref.namelist()
        top_level_dirs = [os.path.join(directory, f.split('/')[0]) for f in extracted_files]
        top_level_dirs = list(set(top_level_dirs))  
    return top_level_dirs[0] if top_level_dirs else directory


if __name__ == "__main__":
    import os
    directory = os.getcwd() # working directory
    zip_path = find_zip_file(directory)
    unzip_folder(zip_path, directory)
