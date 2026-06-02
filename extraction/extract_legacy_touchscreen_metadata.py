# -*- coding: utf-8 -*-

"""
Extract metadata from the legacy Neo/NIX touchscreen dataset.

This extractor is intended for datasets containing:
    neo_rat-*_*.pkl
    neo_rat-*_lfps.nio
    neo_rat-*_lfps_clean.nio

Required legacy environment:
    Python 3.7
    neo==0.8.0
    nixio==1.5.1
    numpy==1.20.3
    pandas==1.1.5

This uses the dataset-specific Io.py loader.
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
    Convert Python / NumPy / pandas / quantities / Neo values to JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    # quantities.Quantity
    try:
        import quantities as pq

        if isinstance(value, pq.Quantity):
            try:
                if value.size == 1:
                    return {
                        "value": float(value.magnitude),
                        "unit": str(value.units),
                    }
                return {
                    "value": value.magnitude.tolist(),
                    "unit": str(value.units),
                }
            except Exception:
                return str(value)
    except Exception:
        pass

    # NumPy values
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

    except Exception:
        pass

    # pandas missing values
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    try:
        return str(value)
    except Exception:
        return None


def dataframe_summary(df):
    """
    Return compact metadata about a pandas DataFrame.
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
        "columns": list(df.columns),
    }


def dataframe_records(df, index_name):
    """
    Convert a dataframe to JSON-safe row records, keeping the index as a column.

    This is important because unit_metanames/trial_metanames/lfp_metanames are only
    the names of metadata fields. The actual metadata values are inside the dataframes.
    """

    if df is None:
        return []

    if df.shape[0] == 0:
        return []

    df_copy = df.copy()
    df_copy[index_name] = df_copy.index.astype(str)

    records = df_copy.to_dict(orient="records")

    return make_json_safe(records)


def summarize_numeric_column(df, column_name):
    """
    Summarize a numeric dataframe column.
    """

    if df is None:
        return None

    if column_name not in df.columns:
        return None

    values = pd.to_numeric(df[column_name], errors="coerce").dropna()

    if len(values) == 0:
        return None

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def summarize_dataframe_values(df, columns):
    """
    Extract unique values from selected dataframe columns.
    """

    output = {}

    if df is None:
        return output

    for column in columns:
        if column in df.columns:
            values = df[column].dropna().astype(str).unique().tolist()
            output[column] = sorted(values)

    return output


def summarize_event_counts(segment):
    """
    Count events by Neo Event object name.
    """

    event_counts = {}

    for event in segment.events:
        name = event.name

        if name is None:
            name = "unnamed_event"

        event_counts[name] = len(event)

    return event_counts


def summarize_epoch_counts(segment):
    """
    Count epochs by Neo Epoch object name.
    """

    epoch_counts = {}

    for epoch in segment.epochs:
        name = epoch.name

        if name is None:
            name = "unnamed_epoch"

        epoch_counts[name] = len(epoch)

    return epoch_counts


def summarize_spikes(segment):
    """
    Summarize spike trains.
    """

    spike_counts = []
    spiketrain_names = []

    for spiketrain in segment.spiketrains:
        spike_counts.append(len(spiketrain))
        spiketrain_names.append(spiketrain.name)

    if len(spike_counts) == 0:
        return {
            "n_spiketrains": 0,
            "n_spikes_total": 0,
            "min_spikes_per_unit": None,
            "max_spikes_per_unit": None,
            "mean_spikes_per_unit": None,
            "spiketrain_names": [],
        }

    return {
        "n_spiketrains": len(spike_counts),
        "n_spikes_total": int(sum(spike_counts)),
        "min_spikes_per_unit": int(min(spike_counts)),
        "max_spikes_per_unit": int(max(spike_counts)),
        "mean_spikes_per_unit": float(sum(spike_counts) / len(spike_counts)),
        "spiketrain_names": make_json_safe(spiketrain_names),
    }


def summarize_analogsignals(segment):
    """
    Summarize analog signals, including LFP signals when loaded.
    """

    signals = []

    for signal in segment.analogsignals:
        signal_info = {
            "name": signal.name,
            "shape": list(signal.shape),
            "n_samples": int(signal.shape[0]) if len(signal.shape) > 0 else None,
            "n_channels": int(signal.shape[1]) if len(signal.shape) > 1 else 1,
            "units": str(signal.units),
            "sampling_rate": make_json_safe(getattr(signal, "sampling_rate", None)),
            "t_start": make_json_safe(getattr(signal, "t_start", None)),
            "duration": make_json_safe(getattr(signal, "duration", None)),
        }

        signals.append(signal_info)

    return signals


def extract_block_metadata(block):
    """
    Extract structural metadata from a Neo block.
    """

    metadata = {
        "n_segments": len(block.segments),
        "block_annotations": make_json_safe(dict(block.annotations)),
        "segments": [],
    }

    for segment in block.segments:

        segment_metadata = {
            "name": segment.name,
            "description": segment.description,
            "n_spiketrains": len(segment.spiketrains),
            "n_analogsignals": len(segment.analogsignals),
            "n_events": len(segment.events),
            "n_epochs": len(segment.epochs),
            "spiketrains": [],
            "analogsignals": [],
            "events": [],
            "epochs": [],
        }

        for spiketrain in segment.spiketrains:
            spiketrain_metadata = {
                "name": spiketrain.name,
                "n_spikes": len(spiketrain),
                "t_start": make_json_safe(spiketrain.t_start),
                "t_stop": make_json_safe(spiketrain.t_stop),
                "units": str(spiketrain.units),
                "annotations": make_json_safe(dict(spiketrain.annotations)),
            }

            segment_metadata["spiketrains"].append(spiketrain_metadata)

        for signal in segment.analogsignals:
            signal_metadata = {
                "name": signal.name,
                "shape": list(signal.shape),
                "units": str(signal.units),
                "sampling_rate": make_json_safe(getattr(signal, "sampling_rate", None)),
                "t_start": make_json_safe(getattr(signal, "t_start", None)),
                "duration": make_json_safe(getattr(signal, "duration", None)),
                "annotations": make_json_safe(dict(signal.annotations)),
            }

            segment_metadata["analogsignals"].append(signal_metadata)

        for event in segment.events:
            event_metadata = {
                "name": event.name,
                "n_events": len(event),
                "units": str(event.units),
                "labels_preview": make_json_safe(list(event.labels[:10])) if hasattr(event, "labels") else [],
                "annotations": make_json_safe(dict(event.annotations)),
            }

            segment_metadata["events"].append(event_metadata)

        for epoch in segment.epochs:
            epoch_metadata = {
                "name": epoch.name,
                "n_epochs": len(epoch),
                "units": str(epoch.units),
                "labels_preview": make_json_safe(list(epoch.labels[:10])) if hasattr(epoch, "labels") else [],
                "annotations": make_json_safe(dict(epoch.annotations)),
            }

            segment_metadata["epochs"].append(epoch_metadata)

        metadata["segments"].append(segment_metadata)

    return make_json_safe(metadata)


def extract_session_metadata(io, session_id, read_lfp=False):
    """
    Extract metadata from one session using the legacy Io loader.
    """

    result = {
        "session_id": session_id,
        "animal_id": None,
        "session_date": None,
        "read_lfp": read_lfp,
        "success": False,
        "error": None,
        "traceback": None,

        "summary": None,

        "unit_metanames": [],
        "trial_metanames": [],
        "lfp_metanames": [],
        "lfp_clean_metanames": [],
        "eventnames": [],

        "unit_dataframe": None,
        "trial_dataframe": None,
        "lfp_dataframe": None,
        "lfp_clean_dataframe": None,

        "unit_records": [],
        "trial_records": [],
        "lfp_records": [],
        "lfp_clean_records": [],

        "block_metadata": None,
    }

    try:
        io.load_session(session_id, read_lfp=read_lfp)

        result["animal_id"] = io.animal_id
        result["session_date"] = io.session_date

        result["unit_metanames"] = make_json_safe(io.unit_metanames)
        result["trial_metanames"] = make_json_safe(io.trial_metanames)
        result["lfp_metanames"] = make_json_safe(io.lfp_metanames)
        result["lfp_clean_metanames"] = make_json_safe(io.lfp_clean_metanames)
        result["eventnames"] = make_json_safe(io.eventnames)

        result["unit_dataframe"] = dataframe_summary(io.unit_df)
        result["trial_dataframe"] = dataframe_summary(io.trial_df)
        result["lfp_dataframe"] = dataframe_summary(io.lfp_df)
        result["lfp_clean_dataframe"] = dataframe_summary(io.lfp_clean_df)

        # Actual metadata values, not just field names.
        result["unit_records"] = dataframe_records(io.unit_df, "unit_id")
        result["trial_records"] = dataframe_records(io.trial_df, "trial_id")
        result["lfp_records"] = dataframe_records(io.lfp_df, "lfp_id")
        result["lfp_clean_records"] = dataframe_records(io.lfp_clean_df, "lfp_clean_id")

        result["block_metadata"] = extract_block_metadata(io.block)

        segment = io.block.segments[0]

        spike_summary = summarize_spikes(segment)
        event_counts = summarize_event_counts(segment)
        epoch_counts = summarize_epoch_counts(segment)
        analogsignal_summary = summarize_analogsignals(segment)

        unit_quality_summary = {
            "firing_rate": summarize_numeric_column(io.unit_df, "firing_rate"),
            "num_spikes": summarize_numeric_column(io.unit_df, "num_spikes"),
            "snr": summarize_numeric_column(io.unit_df, "snr"),
            "isi_violation": summarize_numeric_column(io.unit_df, "isi_violation"),
            "presence_ratio": summarize_numeric_column(io.unit_df, "presence_ratio"),
            "amplitude_cutoff": summarize_numeric_column(io.unit_df, "amplitude_cutoff"),
        }

        unit_categories = summarize_dataframe_values(
            io.unit_df,
            [
                "cluster_group",
                "area",
                "recording_group",
                "celltype",
                "tetrode_area",
                "good_unit",
                "is_artefact",
            ],
        )

        trial_categories = summarize_dataframe_values(
            io.trial_df,
            [
                "modality",
                "correct",
                "object",
                "choice",
                "has_nosetracking",
                "has_whiskertracking",
            ],
        )

        lfp_categories = summarize_dataframe_values(
            io.lfp_df,
            [
                "area",
                "tetrode_area",
                "recording_group",
                "channelname",
                "channel_name",
                "channel_idx",
                "channel",
                "tetrode",
            ],
        )

        result["summary"] = {
            "n_segments": len(io.block.segments),

            "n_spiketrains": len(segment.spiketrains),
            "n_units_from_dataframe": int(io.unit_df.shape[0]),
            "n_unit_metadata_fields": int(io.unit_df.shape[1]),

            "n_spikes_total": spike_summary.get("n_spikes_total"),
            "min_spikes_per_unit": spike_summary.get("min_spikes_per_unit"),
            "max_spikes_per_unit": spike_summary.get("max_spikes_per_unit"),
            "mean_spikes_per_unit": spike_summary.get("mean_spikes_per_unit"),

            "n_analogsignals": len(segment.analogsignals),
            "analogsignals": analogsignal_summary,

            "n_events": len(segment.events),
            "event_counts": event_counts,
            "event_names": list(event_counts.keys()),

            "n_epochs": len(segment.epochs),
            "epoch_counts": epoch_counts,
            "epoch_names": list(epoch_counts.keys()),

            "n_trials_from_dataframe": int(io.trial_df.shape[0]),
            "n_trial_metadata_fields": int(io.trial_df.shape[1]),

            "n_lfp_channels_from_dataframe": int(io.lfp_df.shape[0]),
            "n_lfp_metadata_fields": int(io.lfp_df.shape[1]),

            "unit_quality_summary": unit_quality_summary,
            "unit_categories": unit_categories,
            "trial_categories": trial_categories,
            "lfp_categories": lfp_categories,

            "has_spike_metadata": len(segment.spiketrains) > 0,
            "has_event_metadata": len(segment.events) > 0,
            "has_epoch_metadata": len(segment.epochs) > 0,
            "has_trial_metadata": io.trial_df.shape[0] > 0,
            "has_lfp_metadata": io.lfp_df.shape[0] > 0,
            "has_analogsignals_loaded": len(segment.analogsignals) > 0,
        }

        result["success"] = True

    except Exception as error:
        result["success"] = False
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()

    return make_json_safe(result)


def extract_legacy_touchscreen_dataset(dataset_path, output_folder=None, read_lfp=False):
    """
    Extract metadata from all sessions in the legacy touchscreen dataset.
    """

    dataset_path = Path(dataset_path)

    if output_folder is None:
        output_folder = Path(os.getcwd()) / "outputs" / "extracted_metadata"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    io = Io(path=dataset_path, read_lfp=read_lfp)

    dataset_metadata = {
        "dataset_name": dataset_path.name,
        "dataset_folder": str(dataset_path),
        "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extractor": "legacy_touchscreen_neo08_nixio",
        "read_lfp": read_lfp,
        "environment_note": "Requires Python 3.7, neo==0.8.0, nixio==1.5.1",
        "n_sessions_found": len(io.session_ids),
        "session_ids": io.session_ids,
        "animal_ids": io.animal_ids,
        "sessions": [],
        "dataset_summary": None,
    }

    for session_id in io.session_ids:
        print("Extracting legacy session:", session_id)
        session_metadata = extract_session_metadata(
            io=io,
            session_id=session_id,
            read_lfp=read_lfp,
        )
        dataset_metadata["sessions"].append(session_metadata)

    successful_sessions = [
        session for session in dataset_metadata["sessions"]
        if session.get("success") is True
    ]

    failed_sessions = [
        session for session in dataset_metadata["sessions"]
        if session.get("success") is False
    ]

    dataset_metadata["dataset_summary"] = {
        "n_sessions_found": len(io.session_ids),
        "n_successful_sessions": len(successful_sessions),
        "n_failed_sessions": len(failed_sessions),
        "animal_ids": io.animal_ids,

        "total_spiketrains": sum(
            session["summary"]["n_spiketrains"]
            for session in successful_sessions
            if session.get("summary") is not None
        ),

        "total_spikes": sum(
            session["summary"]["n_spikes_total"]
            for session in successful_sessions
            if session.get("summary") is not None
            and session["summary"].get("n_spikes_total") is not None
        ),

        "total_trials": sum(
            session["summary"]["n_trials_from_dataframe"]
            for session in successful_sessions
            if session.get("summary") is not None
        ),

        "total_lfp_channels": sum(
            session["summary"]["n_lfp_channels_from_dataframe"]
            for session in successful_sessions
            if session.get("summary") is not None
        ),
    }

    output_json = output_folder / (dataset_path.name + "_legacy_touchscreen_metadata.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(dataset_metadata), f, indent=4)

    print("Legacy extraction finished")
    print("Output file:", output_json)

    return dataset_metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract metadata from legacy Neo/NIX touchscreen dataset"
    )

    parser.add_argument(
        "dataset_path",
        help="Path to dataset folder containing neo_rat*.pkl and .nio files"
    )

    parser.add_argument(
        "--read_lfp",
        action="store_true",
        help="Also load LFP .nio files using the custom legacy NixIO loader"
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        help="Output folder"
    )

    args = parser.parse_args()

    extract_legacy_touchscreen_dataset(
        dataset_path=args.dataset_path,
        output_folder=args.output_folder,
        read_lfp=args.read_lfp,
    )