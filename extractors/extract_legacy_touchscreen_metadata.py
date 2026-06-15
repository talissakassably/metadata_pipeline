# -*- coding: utf-8 -*-

"""
Compact metadata extractor for the legacy Neo/NIX touchscreen dataset.

Dataset type:
    neo_rat-*_*.pkl
    neo_rat-*_lfps.nio
    neo_rat-*_lfps_clean.nio

Required legacy environment:
    Python 3.7
    neo==0.8.0
    nixio==1.5.1
    numpy==1.20.3
    pandas==1.1.5

This version is intentionally compact.

It extracts:
    - session id, animal id, date
    - spike train counts and spike summaries
    - event/epoch names and counts
    - trial dataframe summary
    - unit dataframe summary
    - LFP dataframe summary
    - unit quality summaries
    - categorical metadata summaries
    - small dataframe previews
    - array column shapes, not array contents

It does NOT dump:
    - full unit/trial/LFP dataframe records
    - waveform_mean arrays
    - waveform_std arrays
    - template arrays
    - raw analog signal arrays
    - full Neo block object structures

Use --include_records only if you intentionally want full dataframe records.
Even then, array-like columns are summarized by shape instead of dumped.
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


ARRAY_LIKE_COLUMNS = {
    "waveform_mean",
    "waveform_std",
    "template",
    "waveforms",
    "waveform",
    "signal",
    "signals",
    "data",
}


def make_json_safe(value, array_mode="summary"):
    """
    Convert Python / NumPy / pandas / quantities / Neo values to JSON-safe values.

    array_mode:
        "summary" -> arrays become shape/dtype summaries
        "list"    -> arrays become lists, not recommended for big metadata
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
            if array_mode == "list":
                return value.tolist()

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
            try:
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
        return {
            str(key): make_json_safe(item, array_mode=array_mode)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        if len(value) > 100:
            return {
                "list_summary": True,
                "length": len(value),
                "preview": [
                    make_json_safe(item, array_mode=array_mode)
                    for item in list(value)[:5]
                ],
            }

        return [
            make_json_safe(item, array_mode=array_mode)
            for item in value
        ]

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
            "array_like_columns": [],
        }

    array_like_columns = []

    for column in df.columns:
        column_string = str(column)

        if column_string in ARRAY_LIKE_COLUMNS:
            array_like_columns.append(column_string)
            continue

        try:
            first_valid = df[column].dropna().iloc[0]
            if hasattr(first_valid, "shape"):
                array_like_columns.append(column_string)
        except Exception:
            pass

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": [str(column) for column in df.columns],
        "array_like_columns": sorted(list(set(array_like_columns))),
    }


def clean_dataframe_for_records(df):
    """
    Replace heavy array-like dataframe values with compact summaries.
    """

    if df is None:
        return None

    df_copy = df.copy()

    for column in df_copy.columns:
        column_string = str(column)

        if column_string in ARRAY_LIKE_COLUMNS:
            df_copy[column] = df_copy[column].apply(
                lambda value: make_json_safe(value, array_mode="summary")
            )
            continue

        try:
            sample = df_copy[column].dropna().iloc[0]
            if hasattr(sample, "shape"):
                df_copy[column] = df_copy[column].apply(
                    lambda value: make_json_safe(value, array_mode="summary")
                )
        except Exception:
            pass

    return df_copy


def dataframe_preview_records(df, index_name, limit=3):
    """
    Return only a small preview of dataframe rows.
    Array-like values become summaries.
    """

    if df is None:
        return []

    if df.shape[0] == 0:
        return []

    df_copy = clean_dataframe_for_records(df.head(limit))
    df_copy[index_name] = df_copy.index.astype(str)

    return make_json_safe(df_copy.to_dict(orient="records"))


def dataframe_records(df, index_name):
    """
    Return full dataframe records, but with array-like columns summarized.

    This is optional. It can still make output larger, but it will not dump
    waveform/template arrays as millions of JSON lines.
    """

    if df is None:
        return []

    if df.shape[0] == 0:
        return []

    df_copy = clean_dataframe_for_records(df)
    df_copy[index_name] = df_copy.index.astype(str)

    return make_json_safe(df_copy.to_dict(orient="records"))


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
        "n": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def summarize_dataframe_values(df, columns, max_unique_values=30):
    """
    Extract unique values from selected dataframe columns.

    Values are limited to avoid huge JSON output.
    """

    output = {}

    if df is None:
        return output

    for column in columns:
        if column in df.columns:
            try:
                values = df[column].dropna().astype(str).unique().tolist()
                values = sorted(values)

                output[column] = {
                    "n_unique": int(len(values)),
                    "values": values[:max_unique_values],
                    "truncated": len(values) > max_unique_values,
                }
            except Exception as error:
                output[column] = {
                    "error": repr(error),
                }

    return output


def summarize_spikes(segment):
    """
    Summarize spike trains without storing spike times.
    """

    spike_counts = []
    spiketrain_names = []
    annotation_keys = set()

    for spiketrain in segment.spiketrains:
        try:
            spike_counts.append(int(len(spiketrain)))
        except Exception:
            pass

        if getattr(spiketrain, "name", None) is not None:
            spiketrain_names.append(str(spiketrain.name))

        try:
            for key in spiketrain.annotations.keys():
                annotation_keys.add(str(key))
        except Exception:
            pass

    if len(spike_counts) == 0:
        return {
            "n_spiketrains": 0,
            "n_spikes_total": 0,
            "min_spikes_per_unit": None,
            "max_spikes_per_unit": None,
            "mean_spikes_per_unit": None,
            "spiketrain_names_preview": [],
            "spiketrain_annotation_fields": [],
        }

    return {
        "n_spiketrains": int(len(spike_counts)),
        "n_spikes_total": int(sum(spike_counts)),
        "min_spikes_per_unit": int(min(spike_counts)),
        "max_spikes_per_unit": int(max(spike_counts)),
        "mean_spikes_per_unit": float(sum(spike_counts) / len(spike_counts)),
        "spiketrain_names_preview": spiketrain_names[:20],
        "spiketrain_annotation_fields": sorted(list(annotation_keys)),
    }


def summarize_event_counts(segment, max_label_preview=20):
    """
    Count events by Neo Event object name and keep small label previews.
    """

    event_counts = {}
    event_label_preview = {}

    for event in segment.events:
        name = event.name

        if name is None:
            name = "unnamed_event"

        name = str(name)

        try:
            event_counts[name] = int(len(event))
        except Exception:
            event_counts[name] = None

        try:
            labels = [str(label) for label in list(event.labels[:max_label_preview])]
            event_label_preview[name] = labels
        except Exception:
            event_label_preview[name] = []

    return {
        "event_counts": event_counts,
        "event_names": sorted(list(event_counts.keys())),
        "event_label_preview": event_label_preview,
    }


def summarize_epoch_counts(segment, max_label_preview=20):
    """
    Count epochs by Neo Epoch object name and keep small label previews.
    """

    epoch_counts = {}
    epoch_label_preview = {}

    for epoch in segment.epochs:
        name = epoch.name

        if name is None:
            name = "unnamed_epoch"

        name = str(name)

        try:
            epoch_counts[name] = int(len(epoch))
        except Exception:
            epoch_counts[name] = None

        try:
            labels = [str(label) for label in list(epoch.labels[:max_label_preview])]
            epoch_label_preview[name] = labels
        except Exception:
            epoch_label_preview[name] = []

    return {
        "epoch_counts": epoch_counts,
        "epoch_names": sorted(list(epoch_counts.keys())),
        "epoch_label_preview": epoch_label_preview,
    }


def summarize_analogsignals(segment):
    """
    Summarize analog signals/LFP signals without storing data arrays.
    """

    signals = []

    for signal in segment.analogsignals:
        signal_info = {
            "name": getattr(signal, "name", None),
            "shape": list(signal.shape),
            "n_samples": int(signal.shape[0]) if len(signal.shape) > 0 else None,
            "n_channels": int(signal.shape[1]) if len(signal.shape) > 1 else 1,
            "units": str(getattr(signal, "units", None)),
            "sampling_rate": make_json_safe(getattr(signal, "sampling_rate", None)),
            "t_start": make_json_safe(getattr(signal, "t_start", None)),
            "duration": make_json_safe(getattr(signal, "duration", None)),
        }

        signals.append(signal_info)

    return make_json_safe(signals)


def summarize_block_compact(block):
    """
    Compact Neo block summary.

    This avoids saving every spiketrain/event/epoch object.
    """

    block_annotations = {}

    try:
        block_annotations = dict(block.annotations)
    except Exception:
        block_annotations = {}

    segment_summaries = []

    for segment in block.segments:
        segment_summaries.append(
            {
                "name": getattr(segment, "name", None),
                "description": getattr(segment, "description", None),
                "n_spiketrains": int(len(segment.spiketrains)),
                "n_analogsignals": int(len(segment.analogsignals)),
                "n_events": int(len(segment.events)),
                "n_epochs": int(len(segment.epochs)),
            }
        )

    return make_json_safe(
        {
            "n_segments": int(len(block.segments)),
            "block_annotations": block_annotations,
            "segments": segment_summaries,
        }
    )


def extract_session_metadata(
    io,
    session_id,
    read_lfp=False,
    include_records=False,
    preview_limit=3,
    max_unique_values=30,
):
    """
    Extract compact metadata from one session using the legacy Io loader.
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

        "unit_preview_records": [],
        "trial_preview_records": [],
        "lfp_preview_records": [],
        "lfp_clean_preview_records": [],

        "unit_records": None,
        "trial_records": None,
        "lfp_records": None,
        "lfp_clean_records": None,

        "block_metadata_compact": None,
    }

    try:
        io.load_session(session_id, read_lfp=read_lfp)

        result["animal_id"] = make_json_safe(io.animal_id)
        result["session_date"] = make_json_safe(io.session_date)

        result["unit_metanames"] = make_json_safe(io.unit_metanames)
        result["trial_metanames"] = make_json_safe(io.trial_metanames)
        result["lfp_metanames"] = make_json_safe(io.lfp_metanames)
        result["lfp_clean_metanames"] = make_json_safe(io.lfp_clean_metanames)
        result["eventnames"] = make_json_safe(io.eventnames)

        result["unit_dataframe"] = dataframe_summary(io.unit_df)
        result["trial_dataframe"] = dataframe_summary(io.trial_df)
        result["lfp_dataframe"] = dataframe_summary(io.lfp_df)
        result["lfp_clean_dataframe"] = dataframe_summary(io.lfp_clean_df)

        result["unit_preview_records"] = dataframe_preview_records(
            io.unit_df,
            "unit_id",
            limit=preview_limit,
        )
        result["trial_preview_records"] = dataframe_preview_records(
            io.trial_df,
            "trial_id",
            limit=preview_limit,
        )
        result["lfp_preview_records"] = dataframe_preview_records(
            io.lfp_df,
            "lfp_id",
            limit=preview_limit,
        )
        result["lfp_clean_preview_records"] = dataframe_preview_records(
            io.lfp_clean_df,
            "lfp_clean_id",
            limit=preview_limit,
        )

        if include_records:
            result["unit_records"] = dataframe_records(io.unit_df, "unit_id")
            result["trial_records"] = dataframe_records(io.trial_df, "trial_id")
            result["lfp_records"] = dataframe_records(io.lfp_df, "lfp_id")
            result["lfp_clean_records"] = dataframe_records(io.lfp_clean_df, "lfp_clean_id")

        result["block_metadata_compact"] = summarize_block_compact(io.block)

        if len(io.block.segments) == 0:
            raise ValueError("Loaded block contains no segments.")

        segment = io.block.segments[0]

        spike_summary = summarize_spikes(segment)
        event_summary = summarize_event_counts(segment)
        epoch_summary = summarize_epoch_counts(segment)
        analogsignal_summary = summarize_analogsignals(segment)

        unit_quality_summary = {
            "firing_rate": summarize_numeric_column(io.unit_df, "firing_rate"),
            "num_spikes": summarize_numeric_column(io.unit_df, "num_spikes"),
            "snr": summarize_numeric_column(io.unit_df, "snr"),
            "isi_violation": summarize_numeric_column(io.unit_df, "isi_violation"),
            "presence_ratio": summarize_numeric_column(io.unit_df, "presence_ratio"),
            "amplitude_cutoff": summarize_numeric_column(io.unit_df, "amplitude_cutoff"),
            "isolation_distance": summarize_numeric_column(io.unit_df, "isolation_distance"),
            "l_ratio": summarize_numeric_column(io.unit_df, "l_ratio"),
            "d_prime": summarize_numeric_column(io.unit_df, "d_prime"),
            "waveform_amplitude": summarize_numeric_column(io.unit_df, "waveform_amplitude"),
        }

        unit_categories = summarize_dataframe_values(
            io.unit_df,
            [
                "cluster_id",
                "cluster_group",
                "group_id",
                "tetrode",
                "unitid",
                "area",
                "recording_group",
                "celltype",
                "tetrode_area",
                "good_unit",
                "is_artefact",
                "artefact_cause",
            ],
            max_unique_values=max_unique_values,
        )

        trial_categories = summarize_dataframe_values(
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
            max_unique_values=max_unique_values,
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
            max_unique_values=max_unique_values,
        )

        result["summary"] = {
            "n_segments": int(len(io.block.segments)),

            "n_spiketrains": spike_summary.get("n_spiketrains"),
            "n_units_from_dataframe": int(io.unit_df.shape[0]),
            "n_unit_metadata_fields": int(io.unit_df.shape[1]),

            "n_spikes_total": spike_summary.get("n_spikes_total"),
            "min_spikes_per_unit": spike_summary.get("min_spikes_per_unit"),
            "max_spikes_per_unit": spike_summary.get("max_spikes_per_unit"),
            "mean_spikes_per_unit": spike_summary.get("mean_spikes_per_unit"),
            "spiketrain_annotation_fields": spike_summary.get("spiketrain_annotation_fields"),

            "n_analogsignals": int(len(segment.analogsignals)),
            "analogsignals": analogsignal_summary,

            "n_events": int(len(segment.events)),
            "event_counts": event_summary.get("event_counts"),
            "event_names": event_summary.get("event_names"),
            "event_label_preview": event_summary.get("event_label_preview"),

            "n_epochs": int(len(segment.epochs)),
            "epoch_counts": epoch_summary.get("epoch_counts"),
            "epoch_names": epoch_summary.get("epoch_names"),
            "epoch_label_preview": epoch_summary.get("epoch_label_preview"),

            "n_trials_from_dataframe": int(io.trial_df.shape[0]),
            "n_trial_metadata_fields": int(io.trial_df.shape[1]),

            "n_lfp_channels_from_dataframe": int(io.lfp_df.shape[0]),
            "n_lfp_metadata_fields": int(io.lfp_df.shape[1]),

            "n_lfp_clean_channels_from_dataframe": int(io.lfp_clean_df.shape[0]),
            "n_lfp_clean_metadata_fields": int(io.lfp_clean_df.shape[1]),

            "unit_quality_summary": unit_quality_summary,
            "unit_categories": unit_categories,
            "trial_categories": trial_categories,
            "lfp_categories": lfp_categories,

            "has_spike_metadata": spike_summary.get("n_spiketrains", 0) > 0,
            "has_event_metadata": len(segment.events) > 0,
            "has_epoch_metadata": len(segment.epochs) > 0,
            "has_trial_metadata": io.trial_df.shape[0] > 0,
            "has_lfp_metadata": io.lfp_df.shape[0] > 0,
            "has_lfp_clean_metadata": io.lfp_clean_df.shape[0] > 0,
            "has_analogsignals_loaded": len(segment.analogsignals) > 0,
        }

        result["success"] = True

    except Exception as error:
        result["success"] = False
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()

    return make_json_safe(result)


def extract_legacy_touchscreen_dataset(
    dataset_path,
    output_folder=None,
    read_lfp=False,
    include_records=False,
    preview_limit=3,
    max_unique_values=30,
):
    """
    Extract compact metadata from all sessions in the legacy touchscreen dataset.
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
        "extractor": "legacy_touchscreen_compact_neo08_nixio",
        "read_lfp": read_lfp,
        "include_records": include_records,
        "preview_limit": preview_limit,
        "max_unique_values": max_unique_values,
        "environment_note": "Requires Python 3.7, neo==0.8.0, nixio==1.5.1",
        "n_sessions_found": int(len(io.session_ids)),
        "session_ids": make_json_safe(io.session_ids),
        "animal_ids": make_json_safe(io.animal_ids),
        "sessions": [],
        "dataset_summary": None,
    }

    for session_id in io.session_ids:
        print("Extracting legacy session:", session_id)

        session_metadata = extract_session_metadata(
            io=io,
            session_id=session_id,
            read_lfp=read_lfp,
            include_records=include_records,
            preview_limit=preview_limit,
            max_unique_values=max_unique_values,
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

    total_spiketrains = 0
    total_spikes = 0
    total_trials = 0
    total_lfp_channels = 0
    total_lfp_clean_channels = 0

    sessions_with_spikes = 0
    sessions_with_lfp = 0
    sessions_with_trials = 0
    sessions_with_events = 0

    for session in successful_sessions:
        summary = session.get("summary") or {}

        total_spiketrains += summary.get("n_spiketrains") or 0
        total_spikes += summary.get("n_spikes_total") or 0
        total_trials += summary.get("n_trials_from_dataframe") or 0
        total_lfp_channels += summary.get("n_lfp_channels_from_dataframe") or 0
        total_lfp_clean_channels += summary.get("n_lfp_clean_channels_from_dataframe") or 0

        if summary.get("has_spike_metadata") is True:
            sessions_with_spikes += 1

        if summary.get("has_lfp_metadata") is True:
            sessions_with_lfp += 1

        if summary.get("has_trial_metadata") is True:
            sessions_with_trials += 1

        if summary.get("has_event_metadata") is True:
            sessions_with_events += 1

    dataset_metadata["dataset_summary"] = {
        "n_sessions_found": int(len(io.session_ids)),
        "n_successful_sessions": int(len(successful_sessions)),
        "n_failed_sessions": int(len(failed_sessions)),
        "animal_ids": make_json_safe(io.animal_ids),

        "total_spiketrains": int(total_spiketrains),
        "total_spikes": int(total_spikes),
        "total_trials": int(total_trials),
        "total_lfp_channels": int(total_lfp_channels),
        "total_lfp_clean_channels": int(total_lfp_clean_channels),

        "n_sessions_with_spikes": int(sessions_with_spikes),
        "n_sessions_with_lfp_metadata": int(sessions_with_lfp),
        "n_sessions_with_trial_metadata": int(sessions_with_trials),
        "n_sessions_with_event_metadata": int(sessions_with_events),
    }

    output_json = output_folder / (
        dataset_path.name + "_legacy_touchscreen_metadata.json"
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(dataset_metadata), f, indent=2)

    print("Legacy compact extraction finished")
    print("Output file:", output_json)
    print("Successful sessions:", len(successful_sessions))
    print("Failed sessions:", len(failed_sessions))
    print("Total spikes:", total_spikes)
    print("Total trials:", total_trials)
    print("Total LFP channels:", total_lfp_channels)

    return dataset_metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compact metadata extraction from legacy Neo/NIX touchscreen dataset"
    )

    parser.add_argument(
        "dataset_path",
        help="Path to dataset folder containing neo_rat*.pkl and .nio files",
    )

    parser.add_argument(
        "--read_lfp",
        action="store_true",
        help="Also load LFP .nio files using the custom legacy NixIO loader",
    )

    parser.add_argument(
        "--include_records",
        action="store_true",
        help=(
            "Include full unit/trial/LFP dataframe records in JSON. "
            "Array-like values are summarized, but output can still be larger."
        ),
    )

    parser.add_argument(
        "--preview_limit",
        type=int,
        default=3,
        help="Number of dataframe rows to save as preview records. Default: 3",
    )

    parser.add_argument(
        "--max_unique_values",
        type=int,
        default=30,
        help="Maximum unique values saved per categorical metadata field. Default: 30",
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
        include_records=args.include_records,
        preview_limit=args.preview_limit,
        max_unique_values=args.max_unique_values,
    )