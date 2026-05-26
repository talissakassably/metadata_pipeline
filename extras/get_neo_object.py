# -*- coding: utf-8 -*-

""" 
To facilitate data exploration and anotation.

Aim: Returns a neo object
Usage in jupyter notebooks: %run path/to/get_neo_object.py listDatafile

Authors: Alix E. Bonard
Date: 21/10/2024
Last update: 21/10/2024
"""

import io
import warnings
import os 


def get_neo_object(listDatafile):
    """
    input: listDatafile: json file containing a list 


    output: listDataNeo 
    """

    for path_folder, sub_folder, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):

                with open("file_list.json", "r") as f:
                    listDatafile = json.load(f)

                listDataNeo = []

                for path_file in listDatafile:
                    if '.nwb'  not in (path_file):
                        filename = io.get_io(file_or_folder = path_file)
                        listDataNeo.append(filename)
                    elif '.nwb' in path_file:
                        warnings.warn("The package pynwb is needed")
                        filename = io.NWBIO(filename = path_file)
                        listDataNeo.append(filename)

                with open('listDataNeo.json', "w") as f:
                    json.dump(listDataNeo, f)
                print('listDataNeo generated')   
            
    return listDataNeo

if __name__ == "__main__":
    import json 
    import sys

    listDatafile = sys.argv[0] 