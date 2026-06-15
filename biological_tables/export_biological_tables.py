# -*- coding: utf-8 -*-
"""
Biological table export layer.

This is the corrected extraction output layer. It converts heterogeneous JSON
metadata outputs into clean CSV tables used by cross-dataset biological analyses.

Input JSONs:
- openfield metadata JSON
- NWB metadata JSON
- legacy touchscreen metadata JSON
- TouchAndSee metadata JSON

Output CSVs:
- harmonized_biological_sessions.csv
- harmonized_biological_units.csv
- harmonized_biological_trials.csv
- dataset_biological_summary.csv
- extraction_coverage_report.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_num(value, default=np.nan):
    try:
        if value is None or isinstance(value, dict):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None or isinstance(value, dict):
            return default
        return int(float(value))
    except Exception:
        return default


def txt(value):
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def nested(d, keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def classify_context(dataset, description=""):
    description = txt(description).lower()
    if dataset == "openfield_ca1":
        return "open_field"
    if dataset in {"legacy_touchscreen", "touchandsee"}:
        return "object_task"
    if "sleep" in description:
        return "sleep"
    if "sequence" in description or "seq" in description or "figure-eight" in description:
        return "sequence_task"
    if "object" in description or "obj" in description:
        return "open_field_object"
    if "open field" in description or description.strip() == "of":
        return "open_field"
    if description.strip() == "":
        return "unknown"
    return "other"


def add_derived(session_df):
    if session_df.empty:
        return session_df

    for col in [
        "n_units", "n_spikes_total", "n_raw_spike_events", "recording_duration_s",
        "mean_firing_rate_hz", "median_firing_rate_hz", "n_trials",
        "n_events_total", "n_lfp_channels", "n_position_samples", "n_electrodes",
    ]:
        if col not in session_df.columns:
            session_df[col] = np.nan
        session_df[col] = pd.to_numeric(session_df[col], errors="coerce")

    session_df["spikes_per_unit"] = session_df["n_spikes_total"] / session_df["n_units"].replace(0, np.nan)
    session_df["spikes_per_trial"] = session_df["n_spikes_total"] / session_df["n_trials"].replace(0, np.nan)
    session_df["events_per_trial"] = session_df["n_events_total"] / session_df["n_trials"].replace(0, np.nan)
    session_df["units_per_trial"] = session_df["n_units"] / session_df["n_trials"].replace(0, np.nan)

    estimated_fr = (
        session_df["n_spikes_total"]
        / session_df["n_units"].replace(0, np.nan)
        / session_df["recording_duration_s"].replace(0, np.nan)
    )
    session_df["mean_firing_rate_hz_filled"] = session_df["mean_firing_rate_hz"].fillna(estimated_fr)

    session_df["has_units"] = session_df["n_units"].fillna(0) > 0
    session_df["has_spikes"] = session_df["n_spikes_total"].fillna(0) > 0
    session_df["has_trials"] = session_df["n_trials"].fillna(0) > 0
    session_df["has_events"] = session_df["n_events_total"].fillna(0) > 0
    session_df["has_lfp"] = session_df["n_lfp_channels"].fillna(0) > 0
    session_df["has_duration"] = session_df["recording_duration_s"].fillna(0) > 0
    session_df["has_firing_rate"] = session_df["mean_firing_rate_hz_filled"].notna()

    score_cols = ["has_units", "has_spikes", "has_firing_rate", "has_trials", "has_events", "has_lfp"]
    session_df["biological_table_completeness_score"] = session_df[score_cols].mean(axis=1)

    return session_df


def extract_openfield(data):
    sessions, units, trials = [], [], []

    for sess in data.get("sessions", []):
        summary = sess.get("summary", {}) or {}
        neuralynx = sess.get("neuralynx_metadata", {}) or {}
        mclust = sess.get("mclust_metadata", {}) or {}

        session_id = txt(sess.get("session_id"))
        subject_id = txt(sess.get("subject_id"))
        duration = safe_num(summary.get("recording_duration_s", neuralynx.get("recording_duration_s")))
        n_units = safe_int(summary.get("n_units_best_available", mclust.get("sorted_units_mclust")))
        n_sorted_spikes = safe_int(summary.get("sorted_unit_spikes_total_mclust", mclust.get("sorted_unit_spikes_total_mclust")))
        n_raw_spikes = safe_int(summary.get("raw_spike_events_total", neuralynx.get("raw_spike_events_total")))
        n_lfp = safe_int(summary.get("n_lfp_channels", neuralynx.get("n_lfp_channels")))

        mean_fr = n_sorted_spikes / n_units / duration if n_units > 0 and n_sorted_spikes > 0 and duration and duration > 0 else np.nan

        sessions.append({
            "dataset_short_name": "openfield_ca1",
            "session_id": session_id,
            "subject_id": subject_id,
            "session_date": txt(sess.get("session_date")),
            "behavioral_context": "open_field",
            "recording_system": txt(sess.get("recording_system")),
            "brain_region": "CA1",
            "source_file_or_folder": txt(sess.get("relative_session_folder", sess.get("session_folder"))),
            "n_units": n_units,
            "n_spikes_total": n_sorted_spikes,
            "n_raw_spike_events": n_raw_spikes,
            "recording_duration_s": duration,
            "mean_firing_rate_hz": mean_fr,
            "median_firing_rate_hz": np.nan,
            "n_trials": 0,
            "n_events_total": safe_int(neuralynx.get("n_event_records")),
            "n_lfp_channels": n_lfp,
            "n_position_samples": safe_int(neuralynx.get("n_position_records")),
            "n_electrodes": safe_int(neuralynx.get("n_spike_tetrodes")),
            "has_position": as_bool(summary.get("has_position_metadata")),
            "source_level": "session_summary_plus_mclust_units",
            "detail_level": "session_and_units",
        })

        for unit_id, n_spikes in (mclust.get("spikes_per_unit", {}) or {}).items():
            n_spikes = safe_int(n_spikes)
            units.append({
                "dataset_short_name": "openfield_ca1",
                "session_id": session_id,
                "subject_id": subject_id,
                "unit_id": txt(unit_id),
                "brain_region": "CA1",
                "behavioral_context": "open_field",
                "n_spikes": n_spikes,
                "firing_rate_hz": n_spikes / duration if duration and duration > 0 else np.nan,
                "recording_duration_s": duration,
                "cell_type": "",
                "quality_label": "mclust_sorted",
                "is_preview_record": False,
                "source_level": "mclust_spikes_per_unit",
            })

    return pd.DataFrame(sessions), pd.DataFrame(units), pd.DataFrame(trials)


def extract_nwb(data):
    sessions, units, trials = [], [], []

    for item in data.get("files", []):
        nwb = item.get("nwb_extraction", {}) or {}
        meta = item.get("file_metadata", {}) or {}
        if not as_bool(nwb.get("success")):
            continue

        description = txt(nwb.get("session_description"))
        brain_region = "; ".join(txt(x) for x in (nwb.get("electrode_locations", []) or []))

        sessions.append({
            "dataset_short_name": "nwb_ca1",
            "session_id": txt(nwb.get("identifier") or meta.get("file_name")),
            "subject_id": txt(nested(nwb, ["subject", "subject_id"], "")),
            "session_date": txt(nwb.get("session_start_time"))[:10],
            "behavioral_context": classify_context("nwb_ca1", description),
            "raw_session_description": description,
            "recording_system": "NWB / Neuropixels",
            "brain_region": brain_region,
            "source_file_or_folder": txt(meta.get("path")),
            "n_units": safe_int(nwb.get("n_units")),
            "n_spikes_total": np.nan,
            "n_raw_spike_events": np.nan,
            "recording_duration_s": np.nan,
            "mean_firing_rate_hz": np.nan,
            "median_firing_rate_hz": np.nan,
            "n_trials": safe_int(nwb.get("n_trials")),
            "n_events_total": len(nwb.get("intervals", []) or []),
            "n_lfp_channels": 0,
            "n_position_samples": np.nan,
            "n_electrodes": safe_int(nwb.get("n_electrodes")),
            "has_position": "behavior" in (nwb.get("processing_modules", []) or []),
            "source_level": "nwb_session_summary",
            "detail_level": "session_only",
        })

    return pd.DataFrame(sessions), pd.DataFrame(units), pd.DataFrame(trials)


def extract_legacy(data):
    sessions, units, trials = [], [], []

    for sess in data.get("sessions", []):
        summary = sess.get("summary", {}) or {}
        session_id = txt(sess.get("session_id"))
        animal_id = txt(sess.get("animal_id"))
        fr = (summary.get("unit_quality_summary", {}) or {}).get("firing_rate", {}) or {}

        sessions.append({
            "dataset_short_name": "legacy_touchscreen",
            "session_id": session_id,
            "subject_id": animal_id,
            "session_date": txt(sess.get("session_date")),
            "behavioral_context": "object_task",
            "recording_system": "Neuralynx / Neo legacy",
            "brain_region": "VIS; PR; HPC; S1BF",
            "source_file_or_folder": session_id,
            "n_units": safe_int(summary.get("n_units_from_dataframe", summary.get("n_spiketrains"))),
            "n_spikes_total": safe_int(summary.get("n_spikes_total")),
            "n_raw_spike_events": np.nan,
            "recording_duration_s": np.nan,
            "mean_firing_rate_hz": safe_num(fr.get("mean")),
            "median_firing_rate_hz": safe_num(fr.get("median")),
            "n_trials": safe_int(summary.get("n_trials_from_dataframe")),
            "n_events_total": safe_int(sum((summary.get("event_counts", {}) or {}).values())),
            "n_lfp_channels": safe_int(summary.get("n_lfp_channels_from_dataframe")),
            "n_position_samples": np.nan,
            "n_electrodes": np.nan,
            "has_position": False,
            "source_level": "legacy_summary_plus_previews",
            "detail_level": "session_plus_preview_units_trials",
        })

        for rec in sess.get("unit_preview_records", []) or []:
            units.append({
                "dataset_short_name": "legacy_touchscreen",
                "session_id": session_id,
                "subject_id": animal_id,
                "unit_id": txt(rec.get("unit_id", rec.get("unitid"))),
                "brain_region": txt(rec.get("recording_group", rec.get("area"))),
                "behavioral_context": "object_task",
                "n_spikes": safe_int(rec.get("num_spikes")),
                "firing_rate_hz": safe_num(rec.get("firing_rate")),
                "recording_duration_s": np.nan,
                "cell_type": txt(rec.get("celltype")),
                "quality_label": txt(rec.get("cluster_group")),
                "is_preview_record": True,
                "source_level": "legacy_unit_preview_records",
            })

        for rec in sess.get("trial_preview_records", []) or []:
            trials.append({
                "dataset_short_name": "legacy_touchscreen",
                "session_id": session_id,
                "subject_id": animal_id,
                "trial_id": txt(rec.get("trial_id")),
                "trial_type": txt(rec.get("modality")),
                "trial_outcome": txt(rec.get("correct")),
                "trial_side": txt(rec.get("choice")),
                "object_id": txt(rec.get("object")),
                "n_spikes_total": np.nan,
                "n_spiketrains": np.nan,
                "n_events": np.nan,
                "is_preview_record": True,
                "source_level": "legacy_trial_preview_records",
            })

    return pd.DataFrame(sessions), pd.DataFrame(units), pd.DataFrame(trials)


def extract_touchandsee(data):
    sessions, units, trials = [], [], []

    for item in data.get("files", []):
        meta = item.get("file_metadata", {}) or {}
        ext = item.get("touchandsee_extraction", {}) or {}
        obj = ext.get("object_summary", {}) or {}
        if not as_bool(ext.get("success")):
            continue

        session_id = txt(meta.get("session_id") or meta.get("file_name"))
        subject_id = txt(meta.get("subject_id") or meta.get("animal_id"))

        sessions.append({
            "dataset_short_name": "touchandsee",
            "session_id": session_id,
            "subject_id": subject_id,
            "session_date": txt(meta.get("session_date")),
            "behavioral_context": "object_task",
            "recording_system": "Neo pickle legacy",
            "brain_region": "unspecified TouchAndSee recording groups",
            "source_file_or_folder": txt(meta.get("path")),
            "n_units": safe_int(obj.get("n_spiketrains")),
            "n_spikes_total": safe_int(obj.get("n_spikes_total")),
            "n_raw_spike_events": np.nan,
            "recording_duration_s": np.nan,
            "mean_firing_rate_hz": np.nan,
            "median_firing_rate_hz": np.nan,
            "n_trials": safe_int(obj.get("n_segments")),
            "n_events_total": safe_int(obj.get("n_event_times_total")),
            "n_lfp_channels": safe_int(obj.get("n_analogsignals")),
            "n_position_samples": np.nan,
            "n_electrodes": np.nan,
            "has_position": False,
            "source_level": "touchandsee_compact_block_summary",
            "detail_level": "session_plus_preview_trials",
        })

        for seg in ((obj.get("segments", {}) or {}).get("preview", []) or []):
            trials.append({
                "dataset_short_name": "touchandsee",
                "session_id": session_id,
                "subject_id": subject_id,
                "trial_id": txt(seg.get("name")),
                "trial_type": txt(nested(seg, ["annotation_preview", "trial_type"], "")),
                "trial_outcome": txt(nested(seg, ["annotation_preview", "trial_outcome"], "")),
                "trial_side": txt(nested(seg, ["annotation_preview", "trial_side"], "")),
                "object_id": "",
                "n_spikes_total": safe_int(seg.get("n_spikes_total")),
                "n_spiketrains": safe_int(seg.get("n_spiketrains")),
                "n_events": safe_int(seg.get("n_event_times_total")),
                "is_preview_record": True,
                "source_level": "touchandsee_segment_preview",
            })

    return pd.DataFrame(sessions), pd.DataFrame(units), pd.DataFrame(trials)


def summarize(session_df, unit_df, trial_df):
    summary = session_df.groupby("dataset_short_name").agg(
        n_sessions=("session_id", "nunique"),
        n_subjects=("subject_id", "nunique"),
        n_contexts=("behavioral_context", "nunique"),
        median_n_units=("n_units", "median"),
        median_n_spikes_total=("n_spikes_total", "median"),
        median_recording_duration_s=("recording_duration_s", "median"),
        median_firing_rate_hz=("mean_firing_rate_hz_filled", "median"),
        median_n_trials=("n_trials", "median"),
        median_n_events_total=("n_events_total", "median"),
        median_n_lfp_channels=("n_lfp_channels", "median"),
        median_biological_table_completeness_score=("biological_table_completeness_score", "median"),
        sessions_with_units=("has_units", "sum"),
        sessions_with_spikes=("has_spikes", "sum"),
        sessions_with_trials=("has_trials", "sum"),
        sessions_with_events=("has_events", "sum"),
        sessions_with_lfp=("has_lfp", "sum"),
        sessions_with_firing_rate=("has_firing_rate", "sum"),
    ).reset_index()

    coverage_rows = []
    for dataset in sorted(session_df["dataset_short_name"].unique()):
        s = session_df[session_df["dataset_short_name"] == dataset]
        u = unit_df[unit_df["dataset_short_name"] == dataset] if not unit_df.empty and "dataset_short_name" in unit_df.columns else pd.DataFrame()
        t = trial_df[trial_df["dataset_short_name"] == dataset] if not trial_df.empty and "dataset_short_name" in trial_df.columns else pd.DataFrame()
        if not u.empty and "is_preview_record" in u.columns:
            preview_only_units = bool(u["is_preview_record"].fillna(True).all())
        else:
            preview_only_units = True
        coverage_rows.append({
            "dataset_short_name": dataset,
            "n_session_rows": len(s),
            "n_unit_rows": len(u),
            "n_trial_rows": len(t),
            "unit_table_type": "preview_only" if len(u) > 0 and preview_only_units else ("full_or_dictionary_units" if len(u) > 0 else "not_available"),
            "trial_table_type": "preview_only" if len(t) > 0 else "not_available",
            "has_session_level_activity": bool((s["has_units"] | s["has_spikes"]).any()),
            "has_session_level_task": bool((s["has_trials"] | s["has_events"]).any()),
            "has_lfp_availability": bool(s["has_lfp"].any()),
            "median_completeness": float(s["biological_table_completeness_score"].median()),
        })
    coverage = pd.DataFrame(coverage_rows)
    return summary, coverage


def write_report(path, session_df, unit_df, trial_df, summary, coverage):
    lines = [
        "# Biological table export report",
        "",
        "This file confirms that the extraction pipeline produced analysis-ready biological tables.",
        "",
        "## Generated tables",
        f"- Sessions: {len(session_df)} rows",
        f"- Units: {len(unit_df)} rows",
        f"- Trials: {len(trial_df)} rows",
        "",
        "## Coverage",
    ]
    for _, row in coverage.iterrows():
        lines.append(
            f"- {row['dataset_short_name']}: {int(row['n_session_rows'])} sessions, "
            f"{int(row['n_unit_rows'])} unit rows ({row['unit_table_type']}), "
            f"{int(row['n_trial_rows'])} trial rows ({row['trial_table_type']}), "
            f"median completeness={row['median_completeness']:.2f}"
        )
    lines.extend([
        "",
        "## Notes",
        "- Openfield unit rows come from MClust sorted unit spike-count dictionaries.",
        "- NWB currently contributes session-level unit/electrode/context summaries; full per-unit spike-times require a raw NWB table exporter.",
        "- Legacy touchscreen contributes session summaries plus preview units/trials because the compact extractor did not export full records.",
        "- TouchAndSee contributes session summaries plus preview trial segments because the compact extractor did not export all segments.",
        "",
        "These tables are the correct input for the biological cross-dataset case study.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    openfield = load_json(args.openfield_json)
    nwb = load_json(args.nwb_json)
    legacy = load_json(args.legacy_json)
    touchandsee = load_json(args.touchandsee_json)

    o_s, o_u, o_t = extract_openfield(openfield)
    n_s, n_u, n_t = extract_nwb(nwb)
    l_s, l_u, l_t = extract_legacy(legacy)
    t_s, t_u, t_t = extract_touchandsee(touchandsee)

    o_s.to_csv(out / "openfield_sessions.csv", index=False)
    o_u.to_csv(out / "openfield_units.csv", index=False)
    n_s.to_csv(out / "nwb_sessions.csv", index=False)
    l_s.to_csv(out / "legacy_sessions.csv", index=False)
    l_u.to_csv(out / "legacy_units_preview.csv", index=False)
    l_t.to_csv(out / "legacy_trials_preview.csv", index=False)
    t_s.to_csv(out / "touchandsee_sessions.csv", index=False)
    t_t.to_csv(out / "touchandsee_trials_preview.csv", index=False)

    session_df = pd.concat([o_s, n_s, l_s, t_s], ignore_index=True, sort=False)
    unit_df = pd.concat([o_u, n_u, l_u, t_u], ignore_index=True, sort=False)
    trial_df = pd.concat([o_t, n_t, l_t, t_t], ignore_index=True, sort=False)

    session_df = add_derived(session_df)

    session_df.to_csv(out / "harmonized_biological_sessions.csv", index=False)
    unit_df.to_csv(out / "harmonized_biological_units.csv", index=False)
    trial_df.to_csv(out / "harmonized_biological_trials.csv", index=False)

    summary, coverage = summarize(session_df, unit_df, trial_df)
    summary.to_csv(out / "dataset_biological_summary.csv", index=False)
    coverage.to_csv(out / "extraction_coverage_report.csv", index=False)

    write_report(out / "biological_extraction_report.md", session_df, unit_df, trial_df, summary, coverage)

    print("Biological table export complete.")
    print("Output directory:", out)
    print()
    print(summary.to_string(index=False))
    print()
    print(coverage.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Export harmonized biological tables from extracted JSON metadata.")
    parser.add_argument("--openfield_json", required=True)
    parser.add_argument("--nwb_json", required=True)
    parser.add_argument("--legacy_json", required=True)
    parser.add_argument("--touchandsee_json", required=True)
    parser.add_argument("--output_dir", default="outputs/biological_tables")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
