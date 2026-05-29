# -*- coding: utf-8 -*-

"""
Summarize legacy touchscreen metadata JSON.

This creates several CSV files:

1. compact session summary:
   One row per session with internal electrophysiology metadata.

2. metadata fields inventory:
   One row per metadata field name.

3. units detail:
   One row per unit with actual unit metadata values.

4. trials detail:
   One row per trial with actual trial metadata values.

5. LFP detail:
   One row per LFP channel with actual LFP/channel metadata values.

6. LFP clean detail:
   One row per cleaned LFP channel, when available.
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
    Derive spike summary from summary if available,
    otherwise from block_metadata spiketrain details.
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
    Derive event names from summary or block metadata.
    """

    summary = session.get("summary") or {}

    if summary.get("event_names"):
        return summary.get("event_names")

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
    Derive epoch names from summary or block metadata.
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
    Derive compact analog signal/LFP information.
    """

    summary = session.get("summary") or {}

    analogsignals = summary.get("analogsignals")

    if analogsignals is None:
        segment = get_first_segment(session)
        analogsignals = segment.get("analogsignals") or []

    if len(analogsignals) == 0:
        return {
            "analogsignal_names": "",
            "analogsignal_shapes": "",
            "analogsignal_units": "",
            "analogsignal_sampling_rates": "",
            "n_analogsignals_loaded": 0,
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
        "analogsignal_names": " | ".join(names),
        "analogsignal_shapes": " | ".join(shapes),
        "analogsignal_units": " | ".join(units),
        "analogsignal_sampling_rates": " | ".join(sampling_rates),
        "n_analogsignals_loaded": len(analogsignals),
    }


def make_compact_session_summary(metadata):
    """
    Create one compact row per session.
    """

    rows = []

    for session in metadata.get("sessions", []):

        summary = session.get("summary") or {}
        spike_summary = derive_spike_summary(session)
        analog_summary = derive_analogsignal_summary(session)

        event_names = derive_event_names(session)
        epoch_names = derive_epoch_names(session)

        row = {
            "dataset_name": metadata.get("dataset_name"),
            "extractor": metadata.get("extractor"),
            "read_lfp": metadata.get("read_lfp"),

            "session_id": session.get("session_id"),
            "animal_id": session.get("animal_id"),
            "session_date": session.get("session_date"),
            "success": session.get("success"),
            "error": session.get("error"),

            "n_segments": summary.get("n_segments"),
            "n_spiketrains": summary.get("n_spiketrains"),
            "n_units": summary.get("n_units_from_dataframe"),

            "n_spikes_total": spike_summary.get("n_spikes_total"),
            "min_spikes_per_unit": spike_summary.get("min_spikes_per_unit"),
            "max_spikes_per_unit": spike_summary.get("max_spikes_per_unit"),
            "mean_spikes_per_unit": spike_summary.get("mean_spikes_per_unit"),

            "n_events": summary.get("n_events"),
            "event_names": safe_join(event_names),

            "n_epochs": summary.get("n_epochs"),
            "epoch_names": safe_join(epoch_names),

            "n_trials": summary.get("n_trials_from_dataframe"),
            "n_trial_metadata_fields": summary.get("n_trial_metadata_fields"),

            "n_lfp_channels": summary.get("n_lfp_channels_from_dataframe"),
            "n_lfp_metadata_fields": summary.get("n_lfp_metadata_fields"),

            "n_analogsignals": summary.get("n_analogsignals"),
            "n_analogsignals_loaded": analog_summary.get("n_analogsignals_loaded"),
            "analogsignal_names": analog_summary.get("analogsignal_names"),
            "analogsignal_shapes": analog_summary.get("analogsignal_shapes"),
            "analogsignal_units": analog_summary.get("analogsignal_units"),
            "analogsignal_sampling_rates": analog_summary.get("analogsignal_sampling_rates"),

            "n_unit_metadata_fields": summary.get("n_unit_metadata_fields"),

            "has_spike_metadata": summary.get("has_spike_metadata"),
            "has_event_metadata": summary.get("has_event_metadata"),
            "has_epoch_metadata": summary.get("has_epoch_metadata"),
            "has_trial_metadata": summary.get("has_trial_metadata"),
            "has_lfp_metadata": summary.get("has_lfp_metadata"),
            "has_analogsignals_loaded": summary.get("has_analogsignals_loaded"),

            "unit_quality_summary": str(summary.get("unit_quality_summary")),
            "unit_categories": str(summary.get("unit_categories")),
            "trial_categories": str(summary.get("trial_categories")),
            "lfp_categories": str(summary.get("lfp_categories")),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def make_metadata_fields_inventory(metadata):
    """
    Create a long-format inventory of metadata field names.

    This avoids putting huge repeated metaname lists in the main summary CSV.
    """

    rows = []

    for session in metadata.get("sessions", []):

        session_id = session.get("session_id")
        animal_id = session.get("animal_id")
        session_date = session.get("session_date")

        field_groups = {
            "unit_metadata_field": session.get("unit_metanames") or [],
            "trial_metadata_field": session.get("trial_metanames") or [],
            "event_name": session.get("eventnames") or [],
            "lfp_metadata_field": session.get("lfp_metanames") or [],
            "lfp_clean_metadata_field": session.get("lfp_clean_metanames") or [],
        }

        for field_type, fields in field_groups.items():
            for field_name in fields:
                rows.append(
                    {
                        "dataset_name": metadata.get("dataset_name"),
                        "session_id": session_id,
                        "animal_id": animal_id,
                        "session_date": session_date,
                        "field_type": field_type,
                        "field_name": field_name,
                    }
                )

    return pd.DataFrame(rows)


def make_unit_records_table(metadata):
    """
    Create one row per unit with real unit metadata values.
    """

    rows = []

    for session in metadata.get("sessions", []):
        for record in session.get("unit_records") or []:
            row = {
                "dataset_name": metadata.get("dataset_name"),
                "session_id": session.get("session_id"),
                "animal_id": session.get("animal_id"),
                "session_date": session.get("session_date"),
            }
            row.update(record)
            rows.append(row)

    return pd.DataFrame(rows)


def make_trial_records_table(metadata):
    """
    Create one row per trial with real trial metadata values.
    """

    rows = []

    for session in metadata.get("sessions", []):
        for record in session.get("trial_records") or []:
            row = {
                "dataset_name": metadata.get("dataset_name"),
                "session_id": session.get("session_id"),
                "animal_id": session.get("animal_id"),
                "session_date": session.get("session_date"),
            }
            row.update(record)
            rows.append(row)

    return pd.DataFrame(rows)


def make_lfp_records_table(metadata):
    """
    Create one row per LFP channel with real LFP metadata values.
    """

    rows = []

    for session in metadata.get("sessions", []):
        for record in session.get("lfp_records") or []:
            row = {
                "dataset_name": metadata.get("dataset_name"),
                "session_id": session.get("session_id"),
                "animal_id": session.get("animal_id"),
                "session_date": session.get("session_date"),
            }
            row.update(record)
            rows.append(row)

    return pd.DataFrame(rows)


def make_lfp_clean_records_table(metadata):
    """
    Create one row per cleaned LFP channel, when available.
    """

    rows = []

    for session in metadata.get("sessions", []):
        for record in session.get("lfp_clean_records") or []:
            row = {
                "dataset_name": metadata.get("dataset_name"),
                "session_id": session.get("session_id"),
                "animal_id": session.get("animal_id"),
                "session_date": session.get("session_date"),
            }
            row.update(record)
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_legacy_metadata(json_path, output_prefix=None):
    """
    Generate all summary/detail CSV files.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if output_prefix is None:
        base = json_path.replace(".json", "")
    else:
        base = output_prefix

    compact_csv = base + "_compact_session_summary.csv"
    fields_csv = base + "_metadata_fields_inventory.csv"
    units_csv = base + "_units_detail.csv"
    trials_csv = base + "_trials_detail.csv"
    lfp_csv = base + "_lfp_detail.csv"
    lfp_clean_csv = base + "_lfp_clean_detail.csv"

    compact_df = make_compact_session_summary(metadata)
    fields_df = make_metadata_fields_inventory(metadata)
    units_df = make_unit_records_table(metadata)
    trials_df = make_trial_records_table(metadata)
    lfp_df = make_lfp_records_table(metadata)
    lfp_clean_df = make_lfp_clean_records_table(metadata)

    compact_df.to_csv(compact_csv, index=False, encoding="utf-8")
    fields_df.to_csv(fields_csv, index=False, encoding="utf-8")
    units_df.to_csv(units_csv, index=False, encoding="utf-8")
    trials_df.to_csv(trials_csv, index=False, encoding="utf-8")
    lfp_df.to_csv(lfp_csv, index=False, encoding="utf-8")
    lfp_clean_df.to_csv(lfp_clean_csv, index=False, encoding="utf-8")

    print("Compact session summary generated")
    print("Rows:", len(compact_df))
    print("Columns:", len(compact_df.columns))
    print("Output:", compact_csv)

    print("\nMetadata fields inventory generated")
    print("Rows:", len(fields_df))
    print("Columns:", len(fields_df.columns))
    print("Output:", fields_csv)

    print("\nUnit detail table generated")
    print("Rows:", len(units_df))
    print("Columns:", len(units_df.columns))
    print("Output:", units_csv)

    print("\nTrial detail table generated")
    print("Rows:", len(trials_df))
    print("Columns:", len(trials_df.columns))
    print("Output:", trials_csv)

    print("\nLFP detail table generated")
    print("Rows:", len(lfp_df))
    print("Columns:", len(lfp_df.columns))
    print("Output:", lfp_csv)

    print("\nClean LFP detail table generated")
    print("Rows:", len(lfp_clean_df))
    print("Columns:", len(lfp_clean_df.columns))
    print("Output:", lfp_clean_csv)

    print("\nPreview:")
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
        if column in compact_df.columns
    ]

    print(compact_df[available_preview_columns].head())

    return compact_df, fields_df, units_df, trials_df, lfp_df, lfp_clean_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Summarize legacy touchscreen metadata JSON"
    )

    parser.add_argument(
        "json_path",
        help="Path to legacy touchscreen metadata JSON"
    )

    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Optional output prefix without .csv extension"
    )

    args = parser.parse_args()

    summarize_legacy_metadata(
        json_path=args.json_path,
        output_prefix=args.output_prefix,
    )