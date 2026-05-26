# -*- coding: utf-8 -*-

"""
Simple function and script to extract summary information from the files in a
dataset that can be read by Neo.

Usage:

    python extract_from_datafiles path/to/dataset abf

Authors: Andrew P. Davison, Alix E. Bonard
Date: 18/09/2024
Last update: 18/09/2024
"""

import os
from pprint import pprint
from neo import get_io


def walk_dataset(root_dir, filter):
    recording_metadata = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            print(filename)
            path = os.path.join(dirpath, filename)
            if filter(path):
                io = get_io(path)
                data = io.read(lazy=True)[0]
                n_channels_per_segment = set(
                    [
                        sum(sig.shape[1] for sig in seg.analogsignals)
                        for seg in data.segments
                    ]
                )    # Appeler la fonction principale du script

                sampling_rates = set(
                    [
                        sig.sampling_rate.rescale("Hz").item()
                        for seg in data.segments
                        for sig in seg.analogsignals
                    ]
                )
                units = set([
                    sig.units.dimensionality.string for seg in data.segments for sig in seg.analogsignals
                ])
                assert len(n_channels_per_segment) == 1
                (n_channels_per_segment,) = n_channels_per_segment
                assert len(sampling_rates) == 1
                (sampling_rate,) = sampling_rates
                recording_metadata.append(
                    {
                        "path": os.path.relpath(path, root_dir),
                        "n_segments": len(data.segments),
                        "n_channels_per_segment": n_channels_per_segment,
                        "sampling_rate": sampling_rate,
                        "units": units
                    }
                )
    pprint(recording_metadata)
    return recording_metadata


def file_extension_filter(extension):
    def _filter(filepath):
        return filepath.endswith(extension)
    return _filter


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1]
    extension = sys.argv[2]
    walk_dataset(root_dir, file_extension_filter(extension))
