# -*- coding: utf-8 -*-

"""
Compact metadata extractor for the legacy Neo/NIX touchscreen dataset.

This version avoids huge JSON files.

It extracts:
- session identity
- spike/unit counts
- trial counts
- event/epoch names and counts
- LFP channel counts
- unit quality summaries
- compact categorical summaries

It does NOT extract:
- full dataframe records
- waveform arrays
- templates
- raw signal arrays
- full block annotations
- lfpframe from block annotations
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from Io import Io


def make_json_safe(value):
    """
    Convert simple values to JSON-safe objects.
    Arrays are summarized, never expanded.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return str(value)

    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, np.ndarray):
            return {
                "array_summary": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    except Exception:
        pass

    try:
        import quantities as pq

        if isinstance(value, pq.Quantity):
            magnitude = value.magnitude

            try:
                import numpy as np

                if isinstance(magnitude, np.ndarray):
                    if magnitude.size == 1:
                        return {
                            "value": float(magnitude),
                            "unit": str(value.units),
                        }

                    return {
                        "quantity_summary": True,
                        "shape": list(magnitude.shape),
                        "dtype": str(magnitude.dtype),
                        "unit": str(value.units),
                    }
            except Exception:
                pass

            try:
                return {
                    "value": float(magnitude),
                    "unit": str(value.units),
                }
            except Exception:
                return str(value)

    except Exception:
        pass

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, dict):
        output = {}

        for key, item in value.items():
            key_string = str(key)

            # Prevent huge hidden metadata dumps.
            if key_string in [
                "lfpframe",
                "unitframe",
                "trialframe",
                "spikeframe",
                "waveform_mean",
                "waveform_std",
                "template",
                "waveforms",
            ]:
                output[key_string] = "SKIPPED_LARGE_FIELD"
            else:
                output[key_string] = make_json_safe(item)

        return output

    if isinstance(value, (list, tuple, set)):
        value_list = list(value)

        if len(value_list) > 50:
            return {
                "list_summary": True,
                "length": len(value_list),
                "preview": [make_json_safe(item) for item in value_list[:5]],
            }

        return [make_json_safe(item) for item in value_list]

    try:
        return str(value)
    except Exception:
        return None


def dataframe_summary(df):
    """
    Return dataframe row count, column count, and column names only.
    """

    if df is None:
        return {
            "n_rows": 0,
            "n_columns": 0,
            "columns": [],
        }

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": [str(column) for column in df.columns],
    }


def summarize_numeric_column(df, column_name):
    """
    Summarize one numeric dataframe column.
    """

    if df is None:
        return None

    if column_name not in df.columns:
        return None

    try:
        values = pd.to_numeric(df[column_name], errors="coerce").dropna()
    except Exception:
        return None

    if len(values) == 0:
        return None

    return {
        "n": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def summarize_categorical_columns(df, columns, max_values=20):
    """
    Store only limited unique values for selected categorical columns.
    """

    output = {}

    if df is None:
        return output

    for column in columns:
        if column not in df.columns:
            continue

        try:
            values = df[column].dropna().astype(str).unique().tolist()
            values = sorted(values)

            output[column] = {
                "n_unique": int(len(values)),
                "values": values[:max_values],
                "truncated": len(values) > max_values,
            }

        except Exception as error:
            output[column] = {
                "error": repr(error),
            }

    return output


def summarize_spikes(segment):
    """
    Summarize spike trains without saving spike times.
    """

    spike_counts = []
    annotation_fields = set()

    for spiketrain in getattr(segment, "spiketrains", []):
        try:
            spike_counts.append(int(len(spiketrain)))
        except Exception:
            pass

        try:
            for key in spiketrain.annotations.keys():
                if str(key) not in ["waveform_mean", "waveform_std", "template"]:
                    annotation_fields.add(str(key))
        except Exception:
            pass

    if len(spike_counts) == 0:
        return {
            "n_spiketrains": 0,
            "n_spikes_total": 0,
            "min_spikes_per_spiketrain": None,
            "max_spikes_per_spiketrain": None,
            "mean_spikes_per_spiketrain": None,
            "spiketrain_annotation_fields": [],
        }

    return {
        "n_spiketrains": int(len(spike_counts)),
        "n_spikes_total": int(sum(spike_counts)),
        "min_spikes_per_spiketrain": int(min(spike_counts)),
        "max_spikes_per_spiketrain": int(max(spike_counts)),
        "mean_spikes_per_spiketrain": float(sum(spike_counts) / len(spike_counts)),
        "spiketrain_annotation_fields": sorted(list(annotation_fields)),
    }


def summarize_events(segment):
    """
    Summarize event objects by name and count only.
    """

    event_counts = {}

    for event in getattr(segment, "events", []):
        name = getattr(event, "name", None)

        if name is None:
            name = "unnamed_event"

        name = str(name)

        try:
            event_counts[name] = int(len(event))
        except Exception:
            event_counts[name] = None

    return {
        "n_event_objects": int(len(getattr(segment, "events", []))),
        "event_names": sorted(list(event_counts.keys())),
        "event_counts": event_counts,
    }


def summarize_epochs(segment):
    """
    Summarize epoch objects by name and count only.
    """

    epoch_counts = {}

    for epoch in getattr(segment, "epochs", []):
        name = getattr(epoch, "name", None)

        if name is None:
            name = "unnamed_epoch"

        name = str(name)

        try:
            epoch_counts[name] = int(len(epoch))
        except Exception:
            epoch_counts[name] = None

    return {
        "n_epoch_objects": int(len(getattr(segment, "epochs", []))),
        "epoch_names": sorted(list(epoch_counts.keys())),
        "epoch_counts": epoch_counts,
    }


def summarize_analogsignals(segment):
    """
    Summarize analog signal objects by shape only.
    Do not store values.
    """

    signals = []

    for signal in getattr(segment, "analogsignals", []):
        info = {
            "name": make_json_safe(getattr(signal, "name", None)),
            "shape": None,
            "n_samples": None,
            "n_channels": None,
            "units": None,
            "sampling_rate": None,
            "t_start": None,
            "duration": None,
        }

        try:
            info["shape"] = list(signal.shape)
            info["n_samples"] = int(signal.shape[0])
            info["n_channels"] = int(signal.shape[1]) if len(signal.shape) > 1 else 1
        except Exception:
            pass

        try:
            info["units"] = str(signal.units)
        except Exception:
            pass

        try:
            info["sampling_rate"] = make_json_safe(signal.sampling_rate)
        except Exception:
            pass

        try:
            info["t_start"] = make_json_safe(signal.t_start)
        except Exception:
            pass

        try:
            info["duration"] = make_json_safe(signal.duration)
        except Exception:
            pass

        signals.append(info)

    return signals


def extract_one_session(
    io,
    session_id,
    read_lfp=False,
    max_category_values=20,
):
    """
    Extract one compact session summary.
    """

    result = {
        "session_id": session_id,
        "animal_id": None,
        "session_date": None,
        "success": False,
        "error": None,
        "traceback": None,
        "summary": None,
    }

    try:
        io.load_session(session_id, read_lfp=read_lfp)

        result["animal_id"] = make_json_safe(io.animal_id)
        result["session_date"] = make_json_safe(io.session_date)

        if len(io.block.segments) == 0:
            raise ValueError("Loaded block contains no segments.")

        segment = io.block.segments[0]

        spike_summary = summarize_spikes(segment)
        event_summary = summarize_events(segment)
        epoch_summary = summarize_epochs(segment)
        analogsignal_summary = summarize_analogsignals(segment)

        unit_dataframe = dataframe_summary(io.unit_df)
        trial_dataframe = dataframe_summary(io.trial_df)
        lfp_dataframe = dataframe_summary(io.lfp_df)
        lfp_clean_dataframe = dataframe_summary(io.lfp_clean_df)

        unit_quality_summary = {
            "num_spikes": summarize_numeric_column(io.unit_df, "num_spikes"),
            "firing_rate": summarize_numeric_column(io.unit_df, "firing_rate"),
            "snr": summarize_numeric_column(io.unit_df, "snr"),
            "isi_violation": summarize_numeric_column(io.unit_df, "isi_violation"),
            "presence_ratio": summarize_numeric_column(io.unit_df, "presence_ratio"),
            "amplitude_cutoff": summarize_numeric_column(io.unit_df, "amplitude_cutoff"),
            "isolation_distance": summarize_numeric_column(io.unit_df, "isolation_distance"),
            "l_ratio": summarize_numeric_column(io.unit_df, "l_ratio"),
            "d_prime": summarize_numeric_column(io.unit_df, "d_prime"),
            "waveform_amplitude": summarize_numeric_column(io.unit_df, "waveform_amplitude"),
        }

        unit_categories = summarize_categorical_columns(
            io.unit_df,
            [
                "cluster_group",
                "tetrode",
                "area",
                "recording_group",
                "celltype",
                "tetrode_area",
                "good_unit",
                "is_artefact",
            ],
            max_values=max_category_values,
        )

        trial_categories = summarize_categorical_columns(
            io.trial_df,
            [
                "modality",
                "correct",
                "object",
                "choice",
                "block_nr",
                "has_nosetracking",
                "has_whiskertracking",
            ],
            max_values=max_category_values,
        )

        lfp_categories = summarize_categorical_columns(
            io.lfp_df,
            [
                "area",
                "rec_group",
                "recording_group",
                "tetrode",
                "channel",
                "is_bp_filtered",
                "best_on_tetrode",
                "Has invalid timestamps",
            ],
            max_values=max_category_values,
        )

        result["summary"] = {
            "n_segments": int(len(io.block.segments)),

            "n_spiketrains": spike_summary["n_spiketrains"],
            "n_spikes_total": spike_summary["n_spikes_total"],
            "min_spikes_per_spiketrain": spike_summary["min_spikes_per_spiketrain"],
            "max_spikes_per_spiketrain": spike_summary["max_spikes_per_spiketrain"],
            "mean_spikes_per_spiketrain": spike_summary["mean_spikes_per_spiketrain"],
            "spiketrain_annotation_fields": spike_summary["spiketrain_annotation_fields"],

            "n_event_objects": event_summary["n_event_objects"],
            "event_names": event_summary["event_names"],
            "event_counts": event_summary["event_counts"],

            "n_epoch_objects": epoch_summary["n_epoch_objects"],
            "epoch_names": epoch_summary["epoch_names"],
            "epoch_counts": epoch_summary["epoch_counts"],

            "n_analogsignals": int(len(getattr(segment, "analogsignals", []))),
            "analogsignals": analogsignal_summary,

            "unit_dataframe": unit_dataframe,
            "trial_dataframe": trial_dataframe,
            "lfp_dataframe": lfp_dataframe,
            "lfp_clean_dataframe": lfp_clean_dataframe,

            "n_units": unit_dataframe["n_rows"],
            "n_unit_metadata_fields": unit_dataframe["n_columns"],

            "n_trials": trial_dataframe["n_rows"],
            "n_trial_metadata_fields": trial_dataframe["n_columns"],

            "n_lfp_channels": lfp_dataframe["n_rows"],
            "n_lfp_metadata_fields": lfp_dataframe["n_columns"],

            "n_lfp_clean_channels": lfp_clean_dataframe["n_rows"],
            "n_lfp_clean_metadata_fields": lfp_clean_dataframe["n_columns"],

            "unit_quality_summary": unit_quality_summary,
            "unit_categories": unit_categories,
            "trial_categories": trial_categories,
            "lfp_categories": lfp_categories,

            "has_spike_metadata": spike_summary["n_spiketrains"] > 0,
            "has_unit_metadata": unit_dataframe["n_rows"] > 0,
            "has_event_metadata": event_summary["n_event_objects"] > 0,
            "has_epoch_metadata": epoch_summary["n_epoch_objects"] > 0,
            "has_trial_metadata": trial_dataframe["n_rows"] > 0,
            "has_lfp_metadata": lfp_dataframe["n_rows"] > 0,
            "has_lfp_clean_metadata": lfp_clean_dataframe["n_rows"] > 0,
            "has_analogsignals_loaded": len(getattr(segment, "analogsignals", [])) > 0,
        }

        result["success"] = True

    except Exception as error:
        result["success"] = False
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()

    return make_json_safe(result)


def build_dataset_summary(sessions):
    """
    Build compact dataset-level summary.
    """

    successful_sessions = [
        session for session in sessions
        if session.get("success") is True
    ]

    failed_sessions = [
        session for session in sessions
        if session.get("success") is False
    ]

    animal_ids = sorted(
        list(
            set(
                str(session.get("animal_id"))
                for session in successful_sessions
                if session.get("animal_id") is not None
            )
        )
    )

    total_spiketrains = 0
    total_spikes = 0
    total_trials = 0
    total_units = 0
    total_lfp_channels = 0
    total_lfp_clean_channels = 0

    for session in successful_sessions:
        summary = session.get("summary") or {}

        total_spiketrains += summary.get("n_spiketrains") or 0
        total_spikes += summary.get("n_spikes_total") or 0
        total_trials += summary.get("n_trials") or 0
        total_units += summary.get("n_units") or 0
        total_lfp_channels += summary.get("n_lfp_channels") or 0
        total_lfp_clean_channels += summary.get("n_lfp_clean_channels") or 0

    return {
        "n_sessions": int(len(sessions)),
        "n_successful_sessions": int(len(successful_sessions)),
        "n_failed_sessions": int(len(failed_sessions)),
        "animal_ids": animal_ids,
        "n_animals": int(len(animal_ids)),
        "total_spiketrains": int(total_spiketrains),
        "total_spikes": int(total_spikes),
        "total_trials": int(total_trials),
        "total_units": int(total_units),
        "total_lfp_channels": int(total_lfp_channels),
        "total_lfp_clean_channels": int(total_lfp_clean_channels),
    }


def extract_legacy_touchscreen_dataset(
    dataset_path,
    output_folder=None,
    read_lfp=False,
    max_category_values=20,
):
    """
    Extract compact metadata from all sessions.
    """

    dataset_path = Path(dataset_path)

    if output_folder is None:
        output_folder = Path(os.getcwd()) / "outputs" / "extracted_metadata"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    io = Io(path=dataset_path, read_lfp=read_lfp)

    sessions = []

    print("Sessions found:", len(io.session_ids))

    for session_id in io.session_ids:
        print("Extracting legacy session:", session_id)

        session_metadata = extract_one_session(
            io=io,
            session_id=session_id,
            read_lfp=read_lfp,
            max_category_values=max_category_values,
        )

        sessions.append(session_metadata)

    dataset_metadata = {
        "dataset_name": dataset_path.name,
        "dataset_folder": str(dataset_path),
        "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extractor": "legacy_touchscreen_compact_no_block_annotations",
        "read_lfp": bool(read_lfp),
        "max_category_values": int(max_category_values),
        "environment_note": "Requires Python 3.7, neo==0.8.0, nixio==1.5.1",
        "session_ids": [str(session_id) for session_id in io.session_ids],
        "dataset_summary": build_dataset_summary(sessions),
        "sessions": sessions,
    }

    output_json = output_folder / (
        dataset_path.name + "_legacy_touchscreen_metadata.json"
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)

    print("Legacy compact extraction finished")
    print("Output file:", output_json)
    print("Dataset summary:")
    print(dataset_metadata["dataset_summary"])

    return dataset_metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compact metadata extraction from legacy touchscreen dataset"
    )

    parser.add_argument(
        "dataset_path",
        help="Path to dataset folder containing neo_rat*.pkl and .nio files",
    )

    parser.add_argument(
        "--read_lfp",
        action="store_true",
        help="Load LFP .nio files if available. Metadata only, no signal values stored.",
    )

    parser.add_argument(
        "--max_category_values",
        type=int,
        default=20,
        help="Maximum number of category values to store per categorical field. Default: 20",
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        help="Output folder",
    )

    args = parser.parse_args()

    extract_legacy_touchscreen_dataset(
        dataset_path=args.dataset_path,
        output_folder=args.output_folder,
        read_lfp=args.read_lfp,
        max_category_values=args.max_category_values,
    )