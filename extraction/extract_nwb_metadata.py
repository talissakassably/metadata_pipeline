# -*- coding: utf-8 -*-

"""
Extract NWB-specific metadata using pynwb.
"""

from utils import make_json_safe


def extract_nwb_metadata(file_path):
    """
    input:
        file_path: str

    output:
        NWB-specific metadata dictionary
    """

    metadata = {
        "attempted": True,
        "success": False,
        "error": None,
    }

    try:
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(file_path, "r", load_namespaces=True) as io:
            nwbfile = io.read()

            metadata["success"] = True

            metadata["session_description"] = nwbfile.session_description
            metadata["identifier"] = nwbfile.identifier
            metadata["session_start_time"] = str(nwbfile.session_start_time)

            metadata["experimenter"] = list(nwbfile.experimenter) if nwbfile.experimenter else None
            metadata["institution"] = nwbfile.institution
            metadata["lab"] = nwbfile.lab
            metadata["related_publications"] = list(nwbfile.related_publications) if nwbfile.related_publications else None

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

            metadata["n_acquisition_objects"] = len(nwbfile.acquisition)
            metadata["acquisition_objects"] = list(nwbfile.acquisition.keys())

            metadata["n_processing_modules"] = len(nwbfile.processing)
            metadata["processing_modules"] = list(nwbfile.processing.keys())

            metadata["n_devices"] = len(nwbfile.devices)
            metadata["devices"] = list(nwbfile.devices.keys())

            metadata["n_electrode_groups"] = len(nwbfile.electrode_groups)
            metadata["electrode_groups"] = list(nwbfile.electrode_groups.keys())

            if nwbfile.electrodes is not None:
                try:
                    electrodes_df = nwbfile.electrodes.to_dataframe()

                    metadata["n_electrodes"] = len(electrodes_df)
                    metadata["electrode_columns"] = list(electrodes_df.columns)

                    if "location" in electrodes_df.columns:
                        metadata["electrode_locations"] = list(
                            set([str(x) for x in electrodes_df["location"].dropna()])
                        )

                    if "group_name" in electrodes_df.columns:
                        metadata["electrode_group_names"] = list(
                            set([str(x) for x in electrodes_df["group_name"].dropna()])
                        )

                except Exception as error:
                    metadata["electrodes_error"] = str(error)
            else:
                metadata["n_electrodes"] = 0

            if nwbfile.units is not None:
                try:
                    units_df = nwbfile.units.to_dataframe()

                    metadata["n_units"] = len(units_df)
                    metadata["unit_columns"] = list(units_df.columns)

                except Exception as error:
                    metadata["units_error"] = str(error)
            else:
                metadata["n_units"] = 0

            if nwbfile.trials is not None:
                try:
                    trials_df = nwbfile.trials.to_dataframe()

                    metadata["n_trials"] = len(trials_df)
                    metadata["trial_columns"] = list(trials_df.columns)

                except Exception as error:
                    metadata["trials_error"] = str(error)
            else:
                metadata["n_trials"] = 0

            metadata["intervals"] = list(nwbfile.intervals.keys())

    except Exception as error:
        metadata["error"] = str(error)

    return make_json_safe(metadata)