# -*- coding: utf-8 -*-

"""
Extract NWB-specific metadata using PyNWB.

This extractor is the main internal electrophysiology extractor for .nwb files.
It extracts:
    - session and subject metadata
    - devices and electrode groups
    - electrode table metadata
    - unit table metadata
    - spike count summaries
    - trial metadata
    - interval metadata
    - acquisition/processing object summaries
"""

from utils import (
    make_json_safe,
    dataframe_column_summary,
    dataframe_unique_values,
)


def summarize_table(df, table_name):
    """
    Summarize a pandas dataframe table.
    """

    if df is None:
        return {
            "table_name": table_name,
            "n_rows": 0,
            "n_columns": 0,
            "columns": [],
        }

    return {
        "table_name": table_name,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
    }


def summarize_spike_times(units_df):
    """
    Summarize spike_times column from NWB units table.
    """

    if units_df is None:
        return {
            "n_spikes_total": None,
            "min_spikes_per_unit": None,
            "max_spikes_per_unit": None,
            "mean_spikes_per_unit": None,
        }

    if "spike_times" not in units_df.columns:
        return {
            "n_spikes_total": None,
            "min_spikes_per_unit": None,
            "max_spikes_per_unit": None,
            "mean_spikes_per_unit": None,
        }

    spike_counts = []

    for spike_times in units_df["spike_times"]:
        try:
            spike_counts.append(len(spike_times))
        except Exception:
            pass

    if len(spike_counts) == 0:
        return {
            "n_spikes_total": None,
            "min_spikes_per_unit": None,
            "max_spikes_per_unit": None,
            "mean_spikes_per_unit": None,
        }

    return {
        "n_spikes_total": int(sum(spike_counts)),
        "min_spikes_per_unit": int(min(spike_counts)),
        "max_spikes_per_unit": int(max(spike_counts)),
        "mean_spikes_per_unit": float(sum(spike_counts) / len(spike_counts)),
    }


def summarize_value_counts(df, column_name):
    """
    Return counts for unique values in a dataframe column.
    """

    if df is None:
        return {}

    if column_name not in df.columns:
        return {}

    try:
        counts = df[column_name].dropna().astype(str).value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}
    except Exception:
        return {}


def summarize_acquisition_objects(nwbfile):
    """
    Summarize NWB acquisition objects.
    """

    objects = []

    for name, obj in nwbfile.acquisition.items():
        item = {
            "name": name,
            "object_type": type(obj).__name__,
            "neurodata_type": getattr(obj, "neurodata_type", None),
            "description": getattr(obj, "description", None),
        }

        try:
            data = getattr(obj, "data", None)
            if data is not None:
                item["shape"] = list(data.shape)
        except Exception:
            item["shape"] = None

        try:
            rate = getattr(obj, "rate", None)
            item["rate"] = rate
        except Exception:
            item["rate"] = None

        try:
            unit = getattr(obj, "unit", None)
            item["unit"] = unit
        except Exception:
            item["unit"] = None

        objects.append(item)

    return make_json_safe(objects)


def summarize_processing_modules(nwbfile):
    """
    Summarize NWB processing modules and data interfaces.
    """

    modules = []

    for module_name, module in nwbfile.processing.items():
        module_summary = {
            "name": module_name,
            "description": getattr(module, "description", None),
            "data_interfaces": [],
        }

        try:
            for interface_name, interface in module.data_interfaces.items():
                module_summary["data_interfaces"].append({
                    "name": interface_name,
                    "object_type": type(interface).__name__,
                    "neurodata_type": getattr(interface, "neurodata_type", None),
                })
        except Exception:
            pass

        modules.append(module_summary)

    return make_json_safe(modules)


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

            # ---------------------------------------------------------
            # Basic session metadata
            # ---------------------------------------------------------
            metadata["session_description"] = nwbfile.session_description
            metadata["identifier"] = nwbfile.identifier
            metadata["session_start_time"] = str(nwbfile.session_start_time)

            metadata["experimenter"] = list(nwbfile.experimenter) if nwbfile.experimenter else None
            metadata["institution"] = nwbfile.institution
            metadata["lab"] = nwbfile.lab
            metadata["related_publications"] = (
                list(nwbfile.related_publications)
                if nwbfile.related_publications
                else None
            )

            # ---------------------------------------------------------
            # Subject metadata
            # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # Acquisition and processing
            # ---------------------------------------------------------
            metadata["n_acquisition_objects"] = len(nwbfile.acquisition)
            metadata["acquisition_objects"] = list(nwbfile.acquisition.keys())
            metadata["acquisition_object_summaries"] = summarize_acquisition_objects(nwbfile)

            metadata["n_processing_modules"] = len(nwbfile.processing)
            metadata["processing_modules"] = list(nwbfile.processing.keys())
            metadata["processing_module_summaries"] = summarize_processing_modules(nwbfile)

            # ---------------------------------------------------------
            # Devices and electrode groups
            # ---------------------------------------------------------
            metadata["n_devices"] = len(nwbfile.devices)
            metadata["devices"] = list(nwbfile.devices.keys())

            metadata["n_electrode_groups"] = len(nwbfile.electrode_groups)
            metadata["electrode_groups"] = list(nwbfile.electrode_groups.keys())

            electrode_group_summaries = []

            for group_name, group in nwbfile.electrode_groups.items():
                electrode_group_summaries.append({
                    "name": group_name,
                    "description": getattr(group, "description", None),
                    "location": getattr(group, "location", None),
                    "device": getattr(getattr(group, "device", None), "name", None),
                })

            metadata["electrode_group_summaries"] = electrode_group_summaries

            # ---------------------------------------------------------
            # Electrodes table
            # ---------------------------------------------------------
            metadata["n_electrodes"] = 0
            metadata["electrode_columns"] = []
            metadata["electrode_locations"] = []
            metadata["electrode_group_names"] = []
            metadata["electrode_location_counts"] = {}
            metadata["electrode_group_counts"] = {}

            if nwbfile.electrodes is not None:
                try:
                    electrodes_df = nwbfile.electrodes.to_dataframe()

                    metadata["electrodes_table"] = summarize_table(electrodes_df, "electrodes")
                    metadata["n_electrodes"] = len(electrodes_df)
                    metadata["electrode_columns"] = list(electrodes_df.columns)

                    if "location" in electrodes_df.columns:
                        metadata["electrode_locations"] = dataframe_unique_values(electrodes_df, "location")
                        metadata["electrode_location_counts"] = summarize_value_counts(electrodes_df, "location")

                    if "group_name" in electrodes_df.columns:
                        metadata["electrode_group_names"] = dataframe_unique_values(electrodes_df, "group_name")
                        metadata["electrode_group_counts"] = summarize_value_counts(electrodes_df, "group_name")

                    for coordinate_column in ["x", "y", "z"]:
                        if coordinate_column in electrodes_df.columns:
                            metadata[f"electrode_{coordinate_column}_summary"] = dataframe_column_summary(
                                electrodes_df,
                                coordinate_column,
                            )

                except Exception as error:
                    metadata["electrodes_error"] = str(error)

            # ---------------------------------------------------------
            # Units table
            # ---------------------------------------------------------
            metadata["n_units"] = 0
            metadata["unit_columns"] = []
            metadata["unit_quality_columns"] = []
            metadata["unit_sampling_rates"] = []
            metadata["cluster_quality_values"] = []
            metadata["n_spikes_total"] = None
            metadata["min_spikes_per_unit"] = None
            metadata["max_spikes_per_unit"] = None
            metadata["mean_spikes_per_unit"] = None

            if nwbfile.units is not None:
                try:
                    units_df = nwbfile.units.to_dataframe()

                    metadata["units_table"] = summarize_table(units_df, "units")
                    metadata["n_units"] = len(units_df)
                    metadata["unit_columns"] = list(units_df.columns)

                    spike_summary = summarize_spike_times(units_df)
                    metadata.update(spike_summary)

                    quality_like_columns = []

                    for column in units_df.columns:
                        lower_column = str(column).lower()
                        if (
                            "quality" in lower_column
                            or "snr" in lower_column
                            or "isi" in lower_column
                            or "amplitude" in lower_column
                            or "firing" in lower_column
                            or "rate" in lower_column
                            or "cluster" in lower_column
                        ):
                            quality_like_columns.append(column)

                    metadata["unit_quality_columns"] = quality_like_columns

                    if "sampling_rate" in units_df.columns:
                        metadata["unit_sampling_rates"] = dataframe_unique_values(units_df, "sampling_rate")

                    if "cluster_quality" in units_df.columns:
                        metadata["cluster_quality_values"] = dataframe_unique_values(units_df, "cluster_quality")
                        metadata["cluster_quality_counts"] = summarize_value_counts(units_df, "cluster_quality")

                    numeric_unit_summaries = {}

                    for column in [
                        "sampling_rate",
                        "firing_rate",
                        "snr",
                        "isi_violation",
                        "amplitude_cutoff",
                    ]:
                        summary = dataframe_column_summary(units_df, column)
                        if summary is not None:
                            numeric_unit_summaries[column] = summary

                    metadata["numeric_unit_summaries"] = numeric_unit_summaries

                except Exception as error:
                    metadata["units_error"] = str(error)

            # ---------------------------------------------------------
            # Trials table
            # ---------------------------------------------------------
            metadata["n_trials"] = 0
            metadata["trial_columns"] = []

            if nwbfile.trials is not None:
                try:
                    trials_df = nwbfile.trials.to_dataframe()

                    metadata["trials_table"] = summarize_table(trials_df, "trials")
                    metadata["n_trials"] = len(trials_df)
                    metadata["trial_columns"] = list(trials_df.columns)

                    trial_value_summaries = {}

                    for column in trials_df.columns:
                        if trials_df[column].dtype == object:
                            values = dataframe_unique_values(trials_df, column)
                            if len(values) > 0:
                                trial_value_summaries[column] = values

                    metadata["trial_value_summaries"] = trial_value_summaries

                except Exception as error:
                    metadata["trials_error"] = str(error)

            # ---------------------------------------------------------
            # Intervals
            # ---------------------------------------------------------
            metadata["intervals"] = list(nwbfile.intervals.keys())
            metadata["interval_summaries"] = []

            for interval_name, interval_table in nwbfile.intervals.items():
                interval_summary = {
                    "name": interval_name,
                    "n_rows": None,
                    "columns": [],
                }

                try:
                    interval_df = interval_table.to_dataframe()
                    interval_summary["n_rows"] = len(interval_df)
                    interval_summary["columns"] = list(interval_df.columns)
                except Exception as error:
                    interval_summary["error"] = str(error)

                metadata["interval_summaries"].append(interval_summary)

            # ---------------------------------------------------------
            # Completeness flags
            # ---------------------------------------------------------
            metadata["has_subject_metadata"] = metadata.get("subject") is not None
            metadata["has_electrode_metadata"] = metadata.get("n_electrodes", 0) > 0
            metadata["has_unit_metadata"] = metadata.get("n_units", 0) > 0
            metadata["has_spike_metadata"] = metadata.get("n_spikes_total") is not None
            metadata["has_trial_metadata"] = metadata.get("n_trials", 0) > 0
            metadata["has_interval_metadata"] = len(metadata.get("intervals", [])) > 0

    except Exception as error:
        metadata["error"] = str(error)

    return make_json_safe(metadata)