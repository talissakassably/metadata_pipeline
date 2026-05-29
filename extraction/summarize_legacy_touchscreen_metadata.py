# -*- coding: utf-8 -*-

"""
Summarize legacy touchscreen metadata JSON into one useful CSV.

Output:
    One CSV file with one row per session.

This CSV includes:
    - session/animal metadata
    - spike/unit counts
    - spike count summaries
    - event/epoch counts and names
    - trial counts
    - LFP/channel counts
    - analog signal/LFP loaded information
    - compact summaries of unit, trial, and LFP metadata fields
"""

import json
import argparse
import pandas as pd


def safe_join(value):
    """
    Join list-like values into a readable string.
    """

    if value is None:
        return ""

    if isinstance(value, list):
        return " | ".join([str(item) for item in value])

    return str(value)


def get_first_segment(session):
    """
    Return first segment metadata from block_metadata if available.
    """

    block_metadata = session.get("block_metadata") or {}
    segments = block_metadata.get("segments") or []

    if len(segments) == 0:
        return {}

    return segments[0]


def derive_spike_summary(session):
    """
    Get spike summary from session summary or derive it from block metadata.
    """

    summary = session.get("summary") or {}

    if summary.get("n_spikes_total") is not None:
        return {
            "n_spikes_total": summary.get("n_spikes_total"),
            "min_spikes_per_unit": summary.get("min_spikes_per_unit"),
            "max_spikes_per_unit": summary.get("max_spikes_per_unit"),
            "mean_spikes_per_unit": summary.get("mean_spikes_per_unit"),
        }

    segment = get_first_segment(session)
    spiketrains = segment.get("spiketrains") or []

    spike_counts = []

    for spiketrain in spiketrains:
        n_spikes = spiketrain.get("n_spikes")
        if isinstance(n_spikes, int):
            spike_counts.append(n_spikes)

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


def derive_event_names(session):
    """
    Get event names from summary, eventnames, or block metadata.
    """

    summary = session.get("summary") or {}

    if summary.get("event_names"):
        return summary.get("event_names")

    if session.get("eventnames"):
        return session.get("eventnames")

    segment = get_first_segment(session)
    events = segment.get("events") or []

    names = []

    for event in events:
        name = event.get("name")
        if name is not None and name not in names:
            names.append(name)

    return names


def derive_epoch_names(session):
    """
    Get epoch names from summary or block metadata.
    """

    summary = session.get("summary") or {}

    if summary.get("epoch_names"):
        return summary.get("epoch_names")

    segment = get_first_segment(session)
    epochs = segment.get("epochs") or []

    names = []

    for epoch in epochs:
        name = epoch.get("name")
        if name is not None and name not in names:
            names.append(name)

    return names


def derive_analogsignal_summary(session):
    """
    Get compact analog signal/LFP information.
    """

    summary = session.get("summary") or {}

    analogsignals = summary.get("analogsignals")

    if analogsignals is None:
        segment = get_first_segment(session)
        analogsignals = segment.get("analogsignals") or []

    if len(analogsignals) == 0:
        return {
            "n_analogsignals_loaded": 0,
            "analogsignal_names": "",
            "analogsignal_shapes": "",
            "analogsignal_units": "",
            "analogsignal_sampling_rates": "",
        }

    names = []
    shapes = []
    units = []
    sampling_rates = []

    for signal in analogsignals:
        names.append(str(signal.get("name")))
        shapes.append(str(signal.get("shape")))
        units.append(str(signal.get("units")))
        sampling_rates.append(str(signal.get("sampling_rate")))

    return {
        "n_analogsignals_loaded": len(analogsignals),
        "analogsignal_names": " | ".join(names),
        "analogsignal_shapes": " | ".join(shapes),
        "analogsignal_units": " | ".join(units),
        "analogsignal_sampling_rates": " | ".join(sampling_rates),
    }


def compact_dict(value):
    """
    Convert nested dict summaries into a compact readable string.
    """

    if value is None:
        return ""

    if not isinstance(value, dict):
        return str(value)

    parts = []

    for key, item in value.items():
        if item is None:
            continue
        parts.append(f"{key}: {item}")

    return " | ".join(parts)


def make_one_session_summary(metadata):
    """
    Create one row per session.
    """

    rows = []

    for session in metadata.get("sessions", []):

        summary = session.get("summary") or {}
        spike_summary = derive_spike_summary(session)
        analog_summary = derive_analogsignal_summary(session)

        event_names = derive_event_names(session)
        epoch_names = derive_epoch_names(session)

        unit_dataframe = session.get("unit_dataframe") or {}
        trial_dataframe = session.get("trial_dataframe") or {}
        lfp_dataframe = session.get("lfp_dataframe") or {}

        row = {
            # Dataset/session identity
            "dataset_name": metadata.get("dataset_name"),
            "extractor": metadata.get("extractor"),
            "read_lfp": metadata.get("read_lfp"),

            "session_id": session.get("session_id"),
            "animal_id": session.get("animal_id"),
            "session_date": session.get("session_date"),
            "success": session.get("success"),
            "error": session.get("error"),

            # Core electrophysiology structure
            "n_segments": summary.get("n_segments"),
            "n_spiketrains": summary.get("n_spiketrains"),
            "n_units": summary.get("n_units_from_dataframe") or unit_dataframe.get("n_rows"),

            # Spike summary
            "n_spikes_total": spike_summary.get("n_spikes_total"),
            "min_spikes_per_unit": spike_summary.get("min_spikes_per_unit"),
            "max_spikes_per_unit": spike_summary.get("max_spikes_per_unit"),
            "mean_spikes_per_unit": spike_summary.get("mean_spikes_per_unit"),

            # Events / epochs
            "n_events": summary.get("n_events"),
            "event_names": safe_join(event_names),
            "n_epochs": summary.get("n_epochs"),
            "epoch_names": safe_join(epoch_names),

            # Trials
            "n_trials": summary.get("n_trials_from_dataframe") or trial_dataframe.get("n_rows"),
            "n_trial_metadata_fields": summary.get("n_trial_metadata_fields") or trial_dataframe.get("n_columns"),
            "trial_metadata_fields": safe_join(session.get("trial_metanames")),

            # LFP / analog signals
            "n_lfp_channels": summary.get("n_lfp_channels_from_dataframe") or lfp_dataframe.get("n_rows"),
            "n_lfp_metadata_fields": summary.get("n_lfp_metadata_fields") or lfp_dataframe.get("n_columns"),
            "lfp_metadata_fields": safe_join(session.get("lfp_metanames")),

            "n_analogsignals": summary.get("n_analogsignals"),
            "n_analogsignals_loaded": analog_summary.get("n_analogsignals_loaded"),
            "analogsignal_names": analog_summary.get("analogsignal_names"),
            "analogsignal_shapes": analog_summary.get("analogsignal_shapes"),
            "analogsignal_units": analog_summary.get("analogsignal_units"),
            "analogsignal_sampling_rates": analog_summary.get("analogsignal_sampling_rates"),

            # Unit metadata
            "n_unit_metadata_fields": summary.get("n_unit_metadata_fields") or unit_dataframe.get("n_columns"),
            "unit_metadata_fields": safe_join(session.get("unit_metanames")),

            # Compact category/quality summaries, if available
            "unit_quality_summary": compact_dict(summary.get("unit_quality_summary")),
            "unit_categories": compact_dict(summary.get("unit_categories")),
            "trial_categories": compact_dict(summary.get("trial_categories")),
            "lfp_categories": compact_dict(summary.get("lfp_categories")),

            # Completeness flags
            "has_spike_metadata": summary.get("has_spike_metadata"),
            "has_event_metadata": summary.get("has_event_metadata"),
            "has_epoch_metadata": summary.get("has_epoch_metadata"),
            "has_trial_metadata": summary.get("has_trial_metadata"),
            "has_lfp_metadata": summary.get("has_lfp_metadata"),
            "has_analogsignals_loaded": summary.get("has_analogsignals_loaded"),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_legacy_metadata(json_path, output_csv=None):
    """
    Generate one summary CSV.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df = make_one_session_summary(metadata)

    if output_csv is None:
        output_csv = json_path.replace(".json", "_summary.csv")

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print("Legacy touchscreen summary generated")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Output:", output_csv)

    preview_columns = [
        "session_id",
        "animal_id",
        "success",
        "n_units",
        "n_spikes_total",
        "n_events",
        "n_epochs",
        "n_trials",
        "n_lfp_channels",
        "n_analogsignals_loaded",
    ]

    available_preview_columns = [
        column for column in preview_columns
        if column in df.columns
    ]

    print("\nPreview:")
    print(df[available_preview_columns].head())

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Summarize legacy touchscreen metadata JSON into one CSV"
    )

    parser.add_argument(
        "json_path",
        help="Path to legacy touchscreen metadata JSON"
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional output CSV path"
    )

    args = parser.parse_args()

    summarize_legacy_metadata(
        json_path=args.json_path,
        output_csv=args.output_csv,
    )