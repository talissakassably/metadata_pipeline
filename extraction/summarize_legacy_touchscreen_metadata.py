# -*- coding: utf-8 -*-

import json
import argparse
import pandas as pd


def summarize_legacy_metadata(json_path, output_csv=None):
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    rows = []

    for session in metadata.get("sessions", []):
        summary = session.get("summary") or {}

        row = {
            "session_id": session.get("session_id"),
            "animal_id": session.get("animal_id"),
            "session_date": session.get("session_date"),
            "success": session.get("success"),
            "error": session.get("error"),
            "read_lfp": session.get("read_lfp"),

            "n_segments": summary.get("n_segments"),
            "n_spiketrains": summary.get("n_spiketrains"),
            "n_analogsignals": summary.get("n_analogsignals"),
            "n_events": summary.get("n_events"),
            "n_epochs": summary.get("n_epochs"),

            "n_units_from_dataframe": summary.get("n_units_from_dataframe"),
            "n_unit_metadata_fields": summary.get("n_unit_metadata_fields"),
            "n_trials_from_dataframe": summary.get("n_trials_from_dataframe"),
            "n_trial_metadata_fields": summary.get("n_trial_metadata_fields"),
            "n_lfp_channels_from_dataframe": summary.get("n_lfp_channels_from_dataframe"),
            "n_lfp_metadata_fields": summary.get("n_lfp_metadata_fields"),

            "unit_metanames": " | ".join(session.get("unit_metanames") or []),
            "trial_metanames": " | ".join(session.get("trial_metanames") or []),
            "eventnames": " | ".join(session.get("eventnames") or []),
            "lfp_metanames": " | ".join(session.get("lfp_metanames") or []),
            "n_spikes_total": summary.get("n_spikes_total"),
            "min_spikes_per_unit": summary.get("min_spikes_per_unit"),
            "max_spikes_per_unit": summary.get("max_spikes_per_unit"),
            "mean_spikes_per_unit": summary.get("mean_spikes_per_unit"),

            "event_names_from_summary": " | ".join(summary.get("event_names") or []),
            "epoch_names": " | ".join(summary.get("epoch_names") or []),

            "has_spike_metadata": summary.get("has_spike_metadata"),
            "has_event_metadata": summary.get("has_event_metadata"),
            "has_epoch_metadata": summary.get("has_epoch_metadata"),
            "has_trial_metadata": summary.get("has_trial_metadata"),
            "has_lfp_metadata": summary.get("has_lfp_metadata"),
            "has_analogsignals_loaded": summary.get("has_analogsignals_loaded"),

            "unit_categories": str(summary.get("unit_categories")),
            "trial_categories": str(summary.get("trial_categories")),
            "lfp_categories": str(summary.get("lfp_categories")),
            "unit_quality_summary": str(summary.get("unit_quality_summary")),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_csv is None:
        output_csv = json_path.replace(".json", "_summary.csv")

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print("Summary generated")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Output:", output_csv)

    print("\nDataset-level quick summary:")
    print(df[[
        "session_id",
        "animal_id",
        "success",
        "n_spiketrains",
        "n_analogsignals",
        "n_events",
        "n_epochs",
        "n_units_from_dataframe",
        "n_trials_from_dataframe",
        "n_lfp_channels_from_dataframe"
    ]].head())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    summarize_legacy_metadata(args.json_path, args.output_csv)