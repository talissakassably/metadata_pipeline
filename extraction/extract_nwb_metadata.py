# -*- coding: utf-8 -*-

"""
To facilitate automatic metadata extraction from NWB files.

Aim:
    Extract NWB-specific metadata using pynwb.

This module is used by:
    extract_metadata_pipeline.py

Authors:
    Talissa Kassably

Based on the in-depth metadata extraction workflow.
"""

import os


def extract_nwb_metadata(file_path, root_dir):
    """
    input:
        file_path: str
            path to one NWB file

        root_dir: str
            root folder of the dataset

    output:
        metadata: dict
            extracted NWB-specific metadata
    """

    metadata = {
        "path": os.path.relpath(file_path, root_dir),
        "file_name": os.path.basename(file_path),
        "file_extension": os.path.splitext(file_path)[1],
        "readable_with_pynwb": False,
        "error": None,
    }

    try:
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(file_path, "r", load_namespaces=True) as io:
            nwbfile = io.read()

            metadata["readable_with_pynwb"] = True

            # General NWB file metadata
            metadata["session_description"] = nwbfile.session_description
            metadata["identifier"] = nwbfile.identifier
            metadata["session_start_time"] = str(nwbfile.session_start_time)

            metadata["experimenter"] = list(nwbfile.experimenter) if nwbfile.experimenter else None
            metadata["institution"] = nwbfile.institution
            metadata["lab"] = nwbfile.lab
            metadata["related_publications"] = list(nwbfile.related_publications) if nwbfile.related_publications else None

            # Subject metadata
            if nwbfile.subject is not None:
                metadata["subject"] = {
                    "subject_id": nwbfile.subject.subject_id,
                    "species": nwbfile.subject.species,
                    "sex": nwbfile.subject.sex,
                    "age": nwbfile.subject.age,
                    "description": nwbfile.subject.description,
                    "strain": nwbfile.subject.strain,
                    "genotype": nwbfile.subject.genotype,
                }
            else:
                metadata["subject"] = None

            # Acquisition objects
            metadata["n_acquisition_objects"] = len(nwbfile.acquisition)
            metadata["acquisition_objects"] = list(nwbfile.acquisition.keys())

            # Processing modules
            metadata["n_processing_modules"] = len(nwbfile.processing)
            metadata["processing_modules"] = list(nwbfile.processing.keys())

            # Devices
            metadata["n_devices"] = len(nwbfile.devices)
            metadata["devices"] = list(nwbfile.devices.keys())

            # Electrode groups
            metadata["n_electrode_groups"] = len(nwbfile.electrode_groups)
            metadata["electrode_groups"] = list(nwbfile.electrode_groups.keys())

            # Electrodes table
            if nwbfile.electrodes is not None:
                try:
                    electrodes_df = nwbfile.electrodes.to_dataframe()
                    metadata["n_electrodes"] = len(electrodes_df)

                    if "location" in electrodes_df.columns:
                        metadata["electrode_locations"] = list(
                            set([str(x) for x in electrodes_df["location"].dropna()])
                        )
                    else:
                        metadata["electrode_locations"] = []

                    if "group_name" in electrodes_df.columns:
                        metadata["electrode_group_names"] = list(
                            set([str(x) for x in electrodes_df["group_name"].dropna()])
                        )
                    else:
                        metadata["electrode_group_names"] = []

                except Exception as error:
                    metadata["electrodes_error"] = str(error)
            else:
                metadata["n_electrodes"] = 0
                metadata["electrode_locations"] = []
                metadata["electrode_group_names"] = []

            # Units table, if present
            if nwbfile.units is not None:
                try:
                    units_df = nwbfile.units.to_dataframe()
                    metadata["n_units"] = len(units_df)
                    metadata["unit_columns"] = list(units_df.columns)
                except Exception as error:
                    metadata["units_error"] = str(error)
            else:
                metadata["n_units"] = 0
                metadata["unit_columns"] = []

            # Trials table, if present
            if nwbfile.trials is not None:
                try:
                    trials_df = nwbfile.trials.to_dataframe()
                    metadata["n_trials"] = len(trials_df)
                    metadata["trial_columns"] = list(trials_df.columns)
                except Exception as error:
                    metadata["trials_error"] = str(error)
            else:
                metadata["n_trials"] = 0
                metadata["trial_columns"] = []

            # Intervals
            metadata["intervals"] = list(nwbfile.intervals.keys())

    except Exception as error:
        metadata["error"] = str(error)

    return metadata