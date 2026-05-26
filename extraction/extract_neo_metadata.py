# -*- coding: utf-8 -*-

"""
To facilitate automatic metadata extraction from electrophysiology data files.

Aim:
    Extract summary metadata from one data file readable by Neo.

This module is used by:
    extract_metadata_pipeline.py

Authors:
    Talissa Kassably

Based on scripts by:
    Alix E. Bonard, Andrew P. Davison
"""

import os
from neo import get_io


def read_neo_file(file_path):
    """
    Read one file with Neo.

    Some formats support lazy loading, others do not.
    Therefore, the function first tries lazy=True, then lazy=False.
    """

    io = get_io(file_path)

    try:
        data = io.read(lazy=True)[0]
        loading_mode = "lazy"
    except Exception:
        print("Lazy loading failed, trying without lazy loading:", os.path.basename(file_path))
        data = io.read(lazy=False)[0]
        loading_mode = "non_lazy"

    return io, data, loading_mode


def extract_neo_metadata(file_path, root_dir):
    """
    input:
        file_path: str
            path to one data file readable by Neo

        root_dir: str
            root folder of the dataset

    output:
        metadata: dict
            extracted technical metadata
    """

    metadata = {
        "path": os.path.relpath(file_path, root_dir),
        "file_name": os.path.basename(file_path),
        "file_extension": os.path.splitext(file_path)[1],
        "readable_with_neo": False,
        "loading_mode": None,
        "error": None,
    }

    try:
        io, data, loading_mode = read_neo_file(file_path)

        metadata["readable_with_neo"] = True
        metadata["loading_mode"] = loading_mode
        metadata["neo_io_class"] = io.__class__.__name__
        metadata["n_segments"] = len(data.segments)

        n_analogsignals = 0
        n_spiketrains = 0
        n_events = 0
        n_epochs = 0

        n_channels_per_segment = []
        sampling_rates = []
        units = []
        durations = []

        for segment in data.segments:

            n_analogsignals += len(segment.analogsignals)
            n_spiketrains += len(segment.spiketrains)
            n_events += len(segment.events)
            n_epochs += len(segment.epochs)

            segment_channels = 0

            for signal in segment.analogsignals:

                if len(signal.shape) == 1:
                    n_channels = 1
                else:
                    n_channels = signal.shape[1]

                segment_channels += n_channels

                try:
                    sampling_rates.append(
                        signal.sampling_rate.rescale("Hz").item()
                    )
                except Exception:
                    pass

                try:
                    units.append(
                        signal.units.dimensionality.string
                    )
                except Exception:
                    pass

                try:
                    duration = (signal.t_stop - signal.t_start).rescale("s").item()
                    durations.append(duration)
                except Exception:
                    pass

            n_channels_per_segment.append(segment_channels)

        metadata["n_analogsignals"] = n_analogsignals
        metadata["n_spiketrains"] = n_spiketrains
        metadata["n_events"] = n_events
        metadata["n_epochs"] = n_epochs

        metadata["n_channels_per_segment"] = list(set(n_channels_per_segment))
        metadata["sampling_rates_hz"] = list(set(sampling_rates))
        metadata["units"] = list(set(units))
        metadata["durations_s"] = list(set(durations))

        metadata["has_analogsignals"] = n_analogsignals > 0
        metadata["has_spiketrains"] = n_spiketrains > 0
        metadata["has_events"] = n_events > 0
        metadata["has_epochs"] = n_epochs > 0

    except Exception as error:
        metadata["error"] = str(error)

    return metadata