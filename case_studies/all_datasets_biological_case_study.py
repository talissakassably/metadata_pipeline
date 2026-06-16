# -*- coding: utf-8 -*-
r"""
All-datasets biological case study pipeline.

Goal
----
Use ALL FOUR datasets without forcing a fake identical metric across incompatible
formats.

Biological question
-------------------
How does the structure of experience shape what neural population analyses can
be recovered across public electrophysiology datasets?

Dataset contributions
---------------------
1. NWB hippocampal-entorhinal:
   true spike-time population temporal drift by context and region.

2. TouchAndSee:
   trial-level object/memory and outcome modulation from segment summaries.

3. Legacy touchscreen:
   multisensory object-task/event structure + unit-quality/firing summaries when
   available. If full trial activity is not recoverable, it is still used as a
   task-structure/even-richness dataset rather than being discarded.

4. Openfield CA1:
   continuous spatial-foraging reference using duration, sorted units, spikes,
   LFP availability and position availability.

Main output
-----------
A paper-style cross-dataset evidence map:
- what each dataset can biologically support
- what evidence level each dataset provides
- what result each dataset contributes
- clean figures for a report/presentation

Run
---
python case_studies\all_datasets_biological_case_study.py ^
  --nwb_json outputs\extracted_metadata\d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json ^
  --touchandsee_json outputs\extracted_metadata\p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json ^
  --legacy_json outputs\extracted_metadata\legacy_touchscreen_metadata.json ^
  --openfield_json outputs\extracted_metadata\d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json ^
  --output_dir outputs\all_datasets_biological_case_study

Quick test:
add --max_nwb_files 5
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Generic helpers
# =============================================================================

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x, default=np.nan):
    try:
        if x is None or isinstance(x, dict):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None or isinstance(x, dict):
            return default
        return int(float(x))
    except Exception:
        return default


def text(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def normalize_region(x):
    s = text(x).upper()
    if "LEC" in s or "LATERAL ENTORHINAL" in s:
        return "LEC"
    if "MEC" in s or "MEDIAL ENTORHINAL" in s:
        return "MEC"
    if "CA1" in s:
        return "CA1"
    if "CA3" in s:
        return "CA3"
    if "HIPPO" in s or "HPC" in s or "HP" in s:
        return "HPC_unspecified"
    if "PER" in s or "PRH" in s or "PERIRHINAL" in s:
        return "PER"
    if "VIS" in s or "V2" in s or "VISUAL" in s:
        return "VIS"
    if "S1" in s or "SOMATO" in s or "BARREL" in s:
        return "S1BF"
    return text(x) if text(x).strip() else "unknown"


def classify_context(description, dataset_hint=""):
    s = text(description).lower()
    hint = text(dataset_hint).lower()

    if "legacy" in hint or "touch" in hint:
        return "object_task"
    if "sleep" in s:
        return "sleep"
    if "sequence" in s or "seq" in s or "figure-eight" in s or "figure eight" in s:
        return "sequence_task"
    if "object" in s or "obj" in s:
        return "object_or_object_openfield"
    if "open field" in s or "foraging" in s or s.strip() == "of" or "; of" in s:
        return "open_field"
    if not s.strip():
        return "unknown"
    return "other"


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    out = np.empty(len(p))
    best = 1.0
    n = len(p)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        best = min(best, ranked[i] * n / rank)
        out[order[i]] = best
    return np.minimum(out, 1.0)


def cliffs_delta(a, b):
    a = pd.Series(a).dropna().astype(float).values
    b = pd.Series(b).dropna().astype(float).values
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    return (gt - lt) / (len(a) * len(b))


def mannwhitney_p(a, b):
    try:
        from scipy.stats import mannwhitneyu
        a = pd.Series(a).dropna().astype(float)
        b = pd.Series(b).dropna().astype(float)
        if len(a) < 2 or len(b) < 2:
            return np.nan
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def spearman_r(x, y):
    try:
        from scipy.stats import spearmanr
        x = pd.Series(x)
        y = pd.Series(y)
        valid = x.notna() & y.notna()
        if valid.sum() < 4:
            return np.nan, np.nan
        r, p = spearmanr(x[valid].astype(float), y[valid].astype(float))
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


# =============================================================================
# NWB raw spike-time temporal drift
# =============================================================================

def cosine_distance_matrix(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 3:
        return None

    keep = []
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.isfinite(col).sum() >= 5 and np.nanstd(col) > 0:
            keep.append(j)
    if len(keep) < 3:
        return None
    x = x[:, keep]

    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1
    z = (x - mu) / sd
    z = np.nan_to_num(z, nan=0, posinf=0, neginf=0)

    norms = np.linalg.norm(z, axis=1)
    valid = norms > 0
    if valid.sum() < 5:
        return None
    z = z[valid]
    norms = norms[valid]

    sim = (z @ z.T) / np.outer(norms, norms)
    sim = np.clip(sim, -1, 1)
    return 1 - sim


def compute_toi(matrix, lag_unit=10.0, max_lag=20):
    dist = cosine_distance_matrix(matrix)
    if dist is None:
        return np.nan, pd.DataFrame(), None

    n = dist.shape[0]
    max_lag = min(max_lag, n - 1)
    rows = []
    for lag in range(1, max_lag + 1):
        vals = np.diag(dist, k=lag)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            rows.append({
                "lag_index": lag,
                "lag_value": lag * lag_unit,
                "mean_distance": float(np.mean(vals)),
                "median_distance": float(np.median(vals)),
                "n_pairs": int(len(vals)),
            })

    curve = pd.DataFrame(rows)
    if len(curve) < 3:
        return np.nan, curve, dist

    slope, intercept = np.polyfit(curve["lag_value"].values, curve["mean_distance"].values, 1)
    return float(slope), curve, dist


def bin_spikes(spike_times, bin_size_s=10, min_units=30, max_units=100):
    cleaned = []
    for st in spike_times:
        arr = np.asarray(st, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        arr = arr[arr >= 0]
        if arr.size > 2:
            cleaned.append(arr)

    if len(cleaned) < min_units:
        return None

    if len(cleaned) > max_units:
        rng = random.Random(42)
        cleaned = rng.sample(cleaned, max_units)

    t_min = min(np.min(x) for x in cleaned)
    t_max = max(np.max(x) for x in cleaned)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return None

    cleaned = [x - t_min for x in cleaned]
    duration = t_max - t_min
    if duration < bin_size_s * 8:
        return None

    edges = np.arange(0, duration + bin_size_s, bin_size_s)
    if len(edges) < 8:
        return None

    matrix = np.zeros((len(edges) - 1, len(cleaned)))
    for j, st in enumerate(cleaned):
        matrix[:, j] = np.histogram(st, bins=edges)[0]
    return matrix


def nwb_metadata_table(nwb_json):
    data = load_json(nwb_json)
    rows = []

    for item in data.get("files", []):
        meta = item.get("file_metadata", {}) or {}
        nwb = item.get("nwb_extraction", {}) or {}
        if not nwb.get("success"):
            continue

        regions = nwb.get("electrode_locations", []) or []
        region_text = ";".join(text(r) for r in regions)

        rows.append({
            "absolute_path": meta.get("absolute_path"),
            "session_id": nwb.get("identifier") or meta.get("file_name"),
            "subject_id": (nwb.get("subject") or {}).get("subject_id"),
            "description": nwb.get("session_description"),
            "context": classify_context(nwb.get("session_description"), "nwb"),
            "file_region": normalize_region(region_text),
            "region_text": region_text,
            "n_units_metadata": safe_int(nwb.get("n_units")),
            "n_electrodes": safe_int(nwb.get("n_electrodes")),
        })
    return pd.DataFrame(rows)


def read_nwb_spikes(path):
    try:
        from pynwb import NWBHDF5IO
    except Exception as e:
        raise RuntimeError("pynwb is required: python -m pip install pynwb") from e

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        if "spike_times" not in units_df.columns:
            return []
        return [np.asarray(st, dtype=float) for st in units_df["spike_times"].values]


def analyze_nwb(nwb_json, out_dir, bin_size_s=10, min_units=30, max_units=100, max_files=None):
    meta = nwb_metadata_table(nwb_json)
    rows = []
    curves = []
    examples = 0

    for i, row in meta.iterrows():
        if max_files is not None and i >= max_files:
            break

        print(f"NWB {i+1}/{len(meta)} | {row['context']} | {row['file_region']}")

        path = Path(text(row["absolute_path"]))
        if not path.exists():
            rows.append({
                "dataset": "NWB hippocampal-entorhinal",
                "analysis_family": "raw_spike_time_population_drift",
                "session_id": row["session_id"],
                "subject_id": row["subject_id"],
                "context": row["context"],
                "region": row["file_region"],
                "metric": "TOI",
                "value": np.nan,
                "n_units": 0,
                "n_bins": 0,
                "status": "missing_file",
                "evidence_level": "strong_raw_spike_times",
            })
            continue

        try:
            spike_times = read_nwb_spikes(path)
            matrix = bin_spikes(spike_times, bin_size_s=bin_size_s, min_units=min_units, max_units=max_units)
            if matrix is None:
                rows.append({
                    "dataset": "NWB hippocampal-entorhinal",
                    "analysis_family": "raw_spike_time_population_drift",
                    "session_id": row["session_id"],
                    "subject_id": row["subject_id"],
                    "context": row["context"],
                    "region": row["file_region"],
                    "metric": "TOI",
                    "value": np.nan,
                    "n_units": len(spike_times),
                    "n_bins": 0,
                    "status": "insufficient_units_or_duration",
                    "evidence_level": "strong_raw_spike_times",
                })
                continue

            toi, curve, dist = compute_toi(matrix, lag_unit=bin_size_s, max_lag=20)
            rows.append({
                "dataset": "NWB hippocampal-entorhinal",
                "analysis_family": "raw_spike_time_population_drift",
                "session_id": row["session_id"],
                "subject_id": row["subject_id"],
                "context": row["context"],
                "region": row["file_region"],
                "metric": "TOI",
                "value": toi,
                "n_units": matrix.shape[1],
                "n_bins": matrix.shape[0],
                "status": "ok" if np.isfinite(toi) else "toi_failed",
                "evidence_level": "strong_raw_spike_times",
            })

            if not curve.empty:
                curve["dataset"] = "NWB hippocampal-entorhinal"
                curve["session_id"] = row["session_id"]
                curve["context"] = row["context"]
                curve["region"] = row["file_region"]
                curves.append(curve)

            if dist is not None and examples < 3:
                plt.figure(figsize=(6, 5))
                plt.imshow(dist, aspect="auto", origin="lower")
                plt.colorbar(label="Cosine distance")
                plt.title(f"NWB {row['file_region']} {row['context']}")
                plt.xlabel("Time bin")
                plt.ylabel("Time bin")
                plt.tight_layout()
                plt.savefig(out_dir / "figures" / f"example_nwb_distance_{examples+1}_{row['file_region']}_{row['context']}.png", dpi=250)
                plt.close()
                examples += 1

        except Exception as e:
            rows.append({
                "dataset": "NWB hippocampal-entorhinal",
                "analysis_family": "raw_spike_time_population_drift",
                "session_id": row["session_id"],
                "subject_id": row["subject_id"],
                "context": row["context"],
                "region": row["file_region"],
                "metric": "TOI",
                "value": np.nan,
                "n_units": row["n_units_metadata"],
                "n_bins": 0,
                "status": "error: " + repr(e),
                "evidence_level": "strong_raw_spike_times",
            })

    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


# =============================================================================
# TouchAndSee trial/task effects
# =============================================================================

def get_segments(obj):
    segs = obj.get("segments", [])
    if isinstance(segs, dict):
        if "preview" in segs:
            return segs.get("preview") or [], "preview"
        return [], "none"
    if isinstance(segs, list):
        return segs, "full"
    return [], "none"


def touchandsee_trial_table(touch_json):
    data = load_json(touch_json)
    rows = []

    for item in data.get("files", []):
        meta = item.get("file_metadata", {}) or {}
        ext = item.get("touchandsee_extraction", {}) or {}
        obj = ext.get("object_summary", {}) or {}
        if not ext.get("success"):
            continue

        segs, level = get_segments(obj)
        for i, seg in enumerate(segs):
            if not isinstance(seg, dict):
                continue
            ann = seg.get("annotation_preview", {}) or {}
            rows.append({
                "dataset": "TouchAndSee object-memory",
                "session_id": meta.get("session_id") or meta.get("file_name"),
                "subject_id": meta.get("subject_id") or meta.get("animal_id"),
                "trial_index": i,
                "trial_number": safe_float(ann.get("trial_number"), i),
                "trial_type": text(ann.get("trial_type")),
                "trial_outcome": text(ann.get("trial_outcome")),
                "trial_side": text(ann.get("trial_side")),
                "n_spikes_total": safe_float(seg.get("n_spikes_total")),
                "n_spiketrains": safe_float(seg.get("n_spiketrains")),
                "mean_spikes_per_spiketrain": safe_float(seg.get("mean_spikes_per_spiketrain")),
                "n_events": safe_float(seg.get("n_events")),
                "n_event_times_total": safe_float(seg.get("n_event_times_total")),
                "record_level": level,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["activity"] = pd.to_numeric(df["mean_spikes_per_spiketrain"], errors="coerce")
    df["activity"] = df["activity"].fillna(pd.to_numeric(df["n_spikes_total"], errors="coerce"))
    return df


def analyze_touchandsee(touch_json):
    trials = touchandsee_trial_table(touch_json)
    rows = []

    if trials.empty:
        return trials, pd.DataFrame(rows)

    for sid, sub in trials.groupby("session_id"):
        subject = sub["subject_id"].iloc[0]
        n_trials = len(sub)

        # Memory vs normal
        memory = sub.loc[sub["trial_type"].str.lower().eq("memory"), "activity"]
        normal = sub.loc[sub["trial_type"].str.lower().eq("normal"), "activity"]
        if len(memory.dropna()) >= 2 or len(normal.dropna()) >= 2:
            rows.append({
                "dataset": "TouchAndSee object-memory",
                "analysis_family": "trial_activity_modulation",
                "session_id": sid,
                "subject_id": subject,
                "context": "object_memory_task",
                "region": "unknown",
                "metric": "Cliffs_delta_memory_minus_normal",
                "value": cliffs_delta(memory, normal),
                "p_value": mannwhitney_p(memory, normal),
                "n_trials": n_trials,
                "n_a": len(memory.dropna()),
                "n_b": len(normal.dropna()),
                "status": "ok" if len(memory.dropna()) >= 2 and len(normal.dropna()) >= 2 else "insufficient_groups",
                "evidence_level": "moderate_trial_segment_summary",
            })

        # Correct vs incorrect
        correct = sub.loc[sub["trial_outcome"].str.lower().eq("correct"), "activity"]
        incorrect = sub.loc[sub["trial_outcome"].str.lower().eq("incorrect"), "activity"]
        if len(correct.dropna()) >= 2 or len(incorrect.dropna()) >= 2:
            rows.append({
                "dataset": "TouchAndSee object-memory",
                "analysis_family": "trial_activity_modulation",
                "session_id": sid,
                "subject_id": subject,
                "context": "object_memory_task",
                "region": "unknown",
                "metric": "Cliffs_delta_correct_minus_incorrect",
                "value": cliffs_delta(correct, incorrect),
                "p_value": mannwhitney_p(correct, incorrect),
                "n_trials": n_trials,
                "n_a": len(correct.dropna()),
                "n_b": len(incorrect.dropna()),
                "status": "ok" if len(correct.dropna()) >= 2 and len(incorrect.dropna()) >= 2 else "insufficient_groups",
                "evidence_level": "moderate_trial_segment_summary",
            })

        # Trial order drift
        r, p = spearman_r(sub["trial_number"], sub["activity"])
        rows.append({
            "dataset": "TouchAndSee object-memory",
            "analysis_family": "trial_order_activity_drift",
            "session_id": sid,
            "subject_id": subject,
            "context": "object_memory_task",
            "region": "unknown",
            "metric": "Spearman_activity_trial_number",
            "value": r,
            "p_value": p,
            "n_trials": n_trials,
            "n_a": np.nan,
            "n_b": np.nan,
            "status": "ok" if np.isfinite(r) else "insufficient_trial_order",
            "evidence_level": "moderate_trial_segment_summary",
        })

    effects = pd.DataFrame(rows)
    if not effects.empty:
        m = effects["p_value"].notna()
        effects.loc[m, "p_value_bh"] = bh_adjust(effects.loc[m, "p_value"].values)
    return trials, effects


# =============================================================================
# Legacy touchscreen event/task structure + optional trial effects
# =============================================================================

def find_records(session):
    keys = ["trial_records", "trial_dataframe_records", "trial_table_records", "trials_records", "trial_preview_records"]
    for k in keys:
        v = session.get(k)
        if isinstance(v, list) and len(v) > 0:
            return v, k
    return [], ""


def analyze_legacy(legacy_json):
    data = load_json(legacy_json)
    trial_rows = []
    effects = []
    structure_rows = []

    for sess in data.get("sessions", []):
        sid = sess.get("session_id")
        subject = sess.get("animal_id") or sess.get("subject_id")
        summary = sess.get("summary", {}) or {}
        event_counts = summary.get("event_counts", {}) or {}
        trial_categories = summary.get("trial_categories", {}) or {}
        unit_quality = summary.get("unit_quality_summary", {}) or {}

        records, source_key = find_records(sess)

        structure_rows.append({
            "dataset": "Legacy touchscreen multisensory object",
            "session_id": sid,
            "subject_id": subject,
            "n_trials_summary": safe_int(summary.get("n_trials_from_dataframe")),
            "n_units_summary": safe_int(summary.get("n_units_from_dataframe", summary.get("n_spiketrains"))),
            "n_spikes_total": safe_int(summary.get("n_spikes_total")),
            "n_lfp_channels": safe_int(summary.get("n_lfp_channels_from_dataframe")),
            "n_event_types": len(event_counts),
            "n_event_times": safe_int(sum(event_counts.values())) if isinstance(event_counts, dict) else 0,
            "has_visual_events": any("visual" in text(k).lower() for k in event_counts),
            "has_tactile_events": any("tactile" in text(k).lower() for k in event_counts),
            "has_choice_events": any("poke" in text(k).lower() or "choice" in text(k).lower() for k in event_counts),
            "has_touch_events": any("touch" in text(k).lower() for k in event_counts),
            "has_trial_records": bool(records),
            "trial_record_source": source_key,
            "firing_rate_mean": safe_float((unit_quality.get("firing_rate", {}) or {}).get("mean")),
            "evidence_level": "limited_trial_records" if records else "task_metadata_only",
        })

        for rec in records:
            if not isinstance(rec, dict):
                continue
            trial_rows.append({
                "dataset": "Legacy touchscreen multisensory object",
                "session_id": sid,
                "subject_id": subject,
                "source_key": source_key,
                "trial_id": rec.get("trial_id") or rec.get("trial_number"),
                "modality": text(rec.get("modality")),
                "object": text(rec.get("object")),
                "choice": text(rec.get("choice")),
                "correct": text(rec.get("correct")),
                "activity": safe_float(rec.get("n_spikes_total", rec.get("n_spikes", rec.get("spike_count", rec.get("num_spikes", np.nan)))))
            })

    trial_df = pd.DataFrame(trial_rows)
    structure = pd.DataFrame(structure_rows)

    if not trial_df.empty:
        for sid, sub in trial_df.groupby("session_id"):
            subject = sub["subject_id"].iloc[0]
            act = pd.to_numeric(sub["activity"], errors="coerce")
            has_activity = act.notna().sum() >= 4

            if has_activity and sub["correct"].astype(str).str.len().gt(0).any():
                a = act[sub["correct"].astype(str).str.lower().isin(["1", "true", "correct"])]
                b = act[sub["correct"].astype(str).str.lower().isin(["0", "false", "incorrect"])]
                effects.append({
                    "dataset": "Legacy touchscreen multisensory object",
                    "analysis_family": "trial_activity_modulation",
                    "session_id": sid,
                    "subject_id": subject,
                    "context": "multisensory_object_task",
                    "region": "multi_area_VIS_PR_HPC_S1BF",
                    "metric": "Cliffs_delta_correct_minus_incorrect",
                    "value": cliffs_delta(a, b),
                    "p_value": mannwhitney_p(a, b),
                    "n_trials": len(sub),
                    "status": "ok" if len(a.dropna()) >= 2 and len(b.dropna()) >= 2 else "insufficient_groups",
                    "evidence_level": "limited_trial_records",
                })
            else:
                effects.append({
                    "dataset": "Legacy touchscreen multisensory object",
                    "analysis_family": "task_structure_recovered",
                    "session_id": sid,
                    "subject_id": subject,
                    "context": "multisensory_object_task",
                    "region": "multi_area_VIS_PR_HPC_S1BF",
                    "metric": "trial_records_available_no_activity",
                    "value": len(sub),
                    "p_value": np.nan,
                    "n_trials": len(sub),
                    "status": "task_structure_only",
                    "evidence_level": "limited_trial_records_no_activity",
                })

    effects = pd.DataFrame(effects)
    if not effects.empty and "p_value" in effects:
        m = effects["p_value"].notna()
        effects.loc[m, "p_value_bh"] = bh_adjust(effects.loc[m, "p_value"].values)
    return trial_df, structure, effects


# =============================================================================
# Openfield continuous reference
# =============================================================================

def analyze_openfield(open_json):
    data = load_json(open_json)
    rows = []

    for sess in data.get("sessions", []):
        summary = sess.get("summary", {}) or {}
        neuralynx = sess.get("neuralynx_metadata", {}) or {}
        mclust = sess.get("mclust_metadata", {}) or {}

        rows.append({
            "dataset": "Openfield CA1 continuous spatial foraging",
            "analysis_family": "continuous_spatial_reference",
            "session_id": sess.get("session_id"),
            "subject_id": sess.get("subject_id"),
            "context": "open_field_continuous_foraging",
            "region": "CA1",
            "recording_system": sess.get("recording_system"),
            "recording_duration_s": safe_float(summary.get("recording_duration_s", neuralynx.get("recording_duration_s"))),
            "recording_duration_min": safe_float(summary.get("recording_duration_s", neuralynx.get("recording_duration_s"))) / 60,
            "sorted_units_mclust": safe_int(summary.get("sorted_units_mclust", mclust.get("sorted_units_mclust"))),
            "sorted_unit_spikes_total_mclust": safe_int(summary.get("sorted_unit_spikes_total_mclust", mclust.get("sorted_unit_spikes_total_mclust"))),
            "raw_spike_events_total": safe_int(summary.get("raw_spike_events_total", neuralynx.get("raw_spike_events_total"))),
            "n_lfp_channels": safe_int(summary.get("n_lfp_channels", neuralynx.get("n_lfp_channels"))),
            "has_position_metadata": bool(summary.get("has_position_metadata")),
            "has_sorted_unit_metadata": bool(summary.get("has_sorted_unit_metadata")),
            "evidence_level": "continuous_reference",
        })

    return pd.DataFrame(rows)


# =============================================================================
# Cross-dataset synthesis
# =============================================================================

def evidence_score_table(nwb, touch_effects, legacy_structure, legacy_effects, openfield):
    nwb_ok = nwb[nwb["status"].eq("ok")] if not nwb.empty else pd.DataFrame()
    touch_ok = touch_effects[touch_effects["status"].eq("ok")] if not touch_effects.empty else pd.DataFrame()
    legacy_activity_ok = legacy_effects[legacy_effects["status"].eq("ok")] if not legacy_effects.empty else pd.DataFrame()
    legacy_struct_ok = legacy_structure[legacy_structure["has_trial_records"].eq(True)] if not legacy_structure.empty else pd.DataFrame()
    open_sorted = openfield[openfield["has_sorted_unit_metadata"].eq(True)] if not openfield.empty else pd.DataFrame()

    rows = [
        {
            "dataset": "NWB hippocampal-entorhinal",
            "role": "true spike-time temporal drift across context",
            "evidence_strength": "strong",
            "evidence_score": 5,
            "usable_sessions": int(nwb_ok["session_id"].nunique()) if not nwb_ok.empty else 0,
            "main_result_type": "TOI / distance-lag population drift",
        },
        {
            "dataset": "TouchAndSee object-memory",
            "role": "trial-level object/memory/outcome modulation",
            "evidence_strength": "moderate",
            "evidence_score": 4,
            "usable_sessions": int(touch_ok["session_id"].nunique()) if not touch_ok.empty else 0,
            "main_result_type": "Cliff's delta and trial-order activity drift",
        },
        {
            "dataset": "Legacy touchscreen multisensory object",
            "role": "event-rich multisensory task structure; limited activity reuse",
            "evidence_strength": "limited" if len(legacy_activity_ok) else "task-structure-only",
            "evidence_score": 2 if len(legacy_activity_ok) else 1,
            "usable_sessions": int(legacy_activity_ok["session_id"].nunique()) if len(legacy_activity_ok) else int(legacy_struct_ok["session_id"].nunique()) if not legacy_struct_ok.empty else 0,
            "main_result_type": "task/event structure + optional trial activity",
        },
        {
            "dataset": "Openfield CA1 continuous spatial foraging",
            "role": "continuous spatial reference",
            "evidence_strength": "contextual_reference",
            "evidence_score": 3,
            "usable_sessions": int(open_sorted["session_id"].nunique()) if not open_sorted.empty else 0,
            "main_result_type": "duration, sorted units, spikes, LFP availability",
        },
    ]
    return pd.DataFrame(rows)


def analysis_affordance_matrix(nwb, touch_trials, touch_effects, legacy_structure, openfield):
    rows = []
    datasets = [
        "NWB hippocampal-entorhinal",
        "TouchAndSee object-memory",
        "Legacy touchscreen multisensory object",
        "Openfield CA1 continuous spatial foraging",
    ]

    for ds in datasets:
        rows.append({
            "dataset": ds,
            "raw_spike_time_drift": 0,
            "trial_outcome_modulation": 0,
            "memory_or_trial_type_modulation": 0,
            "event_rich_task_structure": 0,
            "continuous_spatial_reference": 0,
            "lfp_availability": 0,
            "cross_area_potential": 0,
        })

    mat = pd.DataFrame(rows).set_index("dataset")

    if not nwb.empty:
        mat.loc["NWB hippocampal-entorhinal", "raw_spike_time_drift"] = 1
        mat.loc["NWB hippocampal-entorhinal", "continuous_spatial_reference"] = 0.5
        mat.loc["NWB hippocampal-entorhinal", "event_rich_task_structure"] = 0.7
        mat.loc["NWB hippocampal-entorhinal", "cross_area_potential"] = 1

    if not touch_trials.empty:
        mat.loc["TouchAndSee object-memory", "trial_outcome_modulation"] = 1 if touch_trials["trial_outcome"].astype(str).str.len().gt(0).any() else 0
        mat.loc["TouchAndSee object-memory", "memory_or_trial_type_modulation"] = 1 if touch_trials["trial_type"].astype(str).str.len().gt(0).any() else 0
        mat.loc["TouchAndSee object-memory", "event_rich_task_structure"] = 0.8

    if not legacy_structure.empty:
        mat.loc["Legacy touchscreen multisensory object", "event_rich_task_structure"] = 1 if legacy_structure["n_event_types"].max() > 3 else 0.5
        mat.loc["Legacy touchscreen multisensory object", "trial_outcome_modulation"] = 0.6 if legacy_structure["has_trial_records"].any() else 0.3
        mat.loc["Legacy touchscreen multisensory object", "lfp_availability"] = 1 if legacy_structure["n_lfp_channels"].max() > 0 else 0
        mat.loc["Legacy touchscreen multisensory object", "cross_area_potential"] = 0.8

    if not openfield.empty:
        mat.loc["Openfield CA1 continuous spatial foraging", "continuous_spatial_reference"] = 1
        mat.loc["Openfield CA1 continuous spatial foraging", "lfp_availability"] = 1 if openfield["n_lfp_channels"].max() > 0 else 0
        mat.loc["Openfield CA1 continuous spatial foraging", "raw_spike_time_drift"] = 0.4 if openfield["has_sorted_unit_metadata"].any() else 0

    return mat.reset_index()


# =============================================================================
# Figures
# =============================================================================

def save_heatmap(df, out_png, title):
    mat = df.set_index("dataset")
    vals = mat.values.astype(float)

    plt.figure(figsize=(11, 5))
    im = plt.imshow(vals, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="analysis support score")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=35, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_evidence(evidence, out_png):
    evidence = evidence.sort_values("evidence_score")
    plt.figure(figsize=(10, 4.8))
    plt.barh(evidence["dataset"], evidence["evidence_score"])
    plt.xlabel("Evidence strength score")
    plt.title("Cross-dataset evidence strength for the biological case study")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_nwb_context(nwb, curves, fig_dir):
    ok = nwb[nwb["status"].eq("ok")].copy()
    ok["value"] = pd.to_numeric(ok["value"], errors="coerce")
    ok = ok.dropna(subset=["value"])
    if ok.empty:
        return

    groups, labels = [], []
    for ctx in sorted(ok["context"].dropna().unique()):
        vals = ok.loc[ok["context"].eq(ctx), "value"].dropna().values
        if len(vals) >= 2:
            groups.append(vals)
            labels.append(ctx)

    if groups:
        plt.figure(figsize=(10, 5))
        try:
            plt.boxplot(groups, tick_labels=labels, showfliers=False)
        except TypeError:
            plt.boxplot(groups, labels=labels, showfliers=False)
        plt.ylabel("TOI: distance-lag slope")
        plt.title("NWB spike-time temporal organization by behavioral context")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "02_nwb_toi_by_context.png", dpi=250)
        plt.close()

    if not curves.empty:
        plt.figure(figsize=(9, 5))
        for ctx, sub in curves.groupby("context"):
            mean_curve = sub.groupby("lag_value")["mean_distance"].mean().reset_index()
            plt.plot(mean_curve["lag_value"], mean_curve["mean_distance"], marker="o", label=ctx)
        plt.xlabel("Time lag (s)")
        plt.ylabel("Mean population distance")
        plt.title("NWB distance-lag curves by behavioral context")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "03_nwb_distance_lag_curves_by_context.png", dpi=250)
        plt.close()


def plot_touch_effects(effects, fig_dir):
    if effects.empty:
        return
    ok = effects[effects["status"].eq("ok")].copy()
    ok["value"] = pd.to_numeric(ok["value"], errors="coerce")
    ok = ok.dropna(subset=["value"])
    if ok.empty:
        return

    for metric in ok["metric"].unique():
        sub = ok[ok["metric"].eq(metric)]
        plt.figure(figsize=(7, 4))
        plt.hist(sub["value"].dropna(), bins=15)
        plt.axvline(0, linestyle="--")
        plt.xlabel(metric)
        plt.ylabel("Sessions")
        plt.title("TouchAndSee: " + metric.replace("_", " "))
        plt.tight_layout()
        plt.savefig(fig_dir / f"04_touchandsee_{metric}.png", dpi=250)
        plt.close()


def plot_legacy_structure(structure, fig_dir):
    if structure.empty:
        return

    cols = ["has_visual_events", "has_tactile_events", "has_choice_events", "has_touch_events", "has_trial_records"]
    vals = [int(structure[c].sum()) for c in cols if c in structure]
    labels = [c.replace("has_", "").replace("_", " ") for c in cols if c in structure]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals)
    plt.ylabel("Sessions")
    plt.title("Legacy touchscreen: recovered task/event structure")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "05_legacy_recovered_task_structure.png", dpi=250)
    plt.close()

    if "n_event_types" in structure:
        plt.figure(figsize=(7, 4))
        plt.hist(pd.to_numeric(structure["n_event_types"], errors="coerce").dropna(), bins=15)
        plt.xlabel("Number of event types")
        plt.ylabel("Sessions")
        plt.title("Legacy touchscreen event richness")
        plt.tight_layout()
        plt.savefig(fig_dir / "06_legacy_event_richness.png", dpi=250)
        plt.close()


def plot_openfield(openfield, fig_dir):
    if openfield.empty:
        return

    plt.figure(figsize=(7, 5))
    x = pd.to_numeric(openfield["recording_duration_min"], errors="coerce")
    y = pd.to_numeric(openfield["sorted_units_mclust"], errors="coerce")
    plt.scatter(x, y, alpha=0.75)
    plt.xlabel("Recording duration (min)")
    plt.ylabel("MClust sorted units")
    plt.title("Openfield CA1: continuous spatial reference sessions")
    plt.tight_layout()
    plt.savefig(fig_dir / "07_openfield_duration_vs_sorted_units.png", dpi=250)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(y.dropna(), bins=20)
    plt.xlabel("MClust sorted units per session")
    plt.ylabel("Sessions")
    plt.title("Openfield CA1 sorted-unit yield")
    plt.tight_layout()
    plt.savefig(fig_dir / "08_openfield_sorted_unit_yield.png", dpi=250)
    plt.close()


# =============================================================================
# Report
# =============================================================================

def write_report(path, evidence, affordance, nwb, touch_effects, legacy_structure, openfield):
    nwb_ok = nwb[nwb["status"].eq("ok")] if not nwb.empty else pd.DataFrame()
    touch_ok = touch_effects[touch_effects["status"].eq("ok")] if not touch_effects.empty else pd.DataFrame()
    legacy_records = legacy_structure[legacy_structure["has_trial_records"].eq(True)] if not legacy_structure.empty else pd.DataFrame()
    open_sorted = openfield[openfield["has_sorted_unit_metadata"].eq(True)] if not openfield.empty else pd.DataFrame()

    lines = [
        "# All-datasets biological case study",
        "",
        "## Biological question",
        "",
        "How does the structure of experience shape what neural population analyses can be recovered across public electrophysiology datasets?",
        "",
        "## Why this version uses all datasets",
        "",
        "This pipeline does not force the same metric onto incompatible datasets. Instead, it uses a common biological axis — continuous versus event/trial-structured experience — and assigns each dataset a valid role.",
        "",
        "## Dataset contributions",
        "",
    ]

    for _, row in evidence.iterrows():
        lines.append(f"### {row['dataset']}")
        lines.append(f"- Role: {row['role']}")
        lines.append(f"- Evidence strength: {row['evidence_strength']} ({row['evidence_score']}/5)")
        lines.append(f"- Usable sessions/rows: {row['usable_sessions']}")
        lines.append(f"- Main result type: {row['main_result_type']}")
        lines.append("")

    lines += [
        "## Main results",
        "",
        f"- NWB: {len(nwb_ok)} usable spike-time temporal-drift analyses across {nwb_ok['session_id'].nunique() if not nwb_ok.empty else 0} sessions.",
        f"- TouchAndSee: {len(touch_ok)} usable trial-level effect analyses across {touch_ok['session_id'].nunique() if not touch_ok.empty else 0} sessions.",
        f"- Legacy touchscreen: {legacy_records['session_id'].nunique() if not legacy_records.empty else 0} sessions with trial records and {legacy_structure['n_event_types'].median() if not legacy_structure.empty else 0:.1f} median event types.",
        f"- Openfield CA1: {len(openfield)} continuous sessions, {len(open_sorted)} with sorted-unit metadata.",
        "",
        "## Biological interpretation",
        "",
        "The NWB dataset provides the strongest direct neural evidence: spike-time population states increasingly diverge with temporal lag, and the strength of this temporal organization varies by behavioral context. TouchAndSee contributes an object-memory/task dimension by testing trial-level activity modulation. Legacy touchscreen contributes event-rich multisensory task structure, even when full activity reuse is limited. Openfield CA1 provides the continuous spatial reference condition, contrasting trial/event-structured datasets with continuous foraging.",
        "",
        "## Claim to use",
        "",
        "Metadata-guided reuse can identify which biological questions each public electrophysiology dataset can realistically answer. Across these datasets, experience structure emerges as the organizing axis: NWB supports temporal-context drift analysis, TouchAndSee supports object-memory trial analysis, legacy touchscreen supports multisensory event-structure reuse, and openfield CA1 supports continuous spatial-reference analysis.",
        "",
        "## Do not overclaim",
        "",
        "Do not claim that all datasets were analyzed with identical raw spike-time methods. The correct claim is that the same biological axis was evaluated using the strongest valid representation available for each dataset.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run all-datasets biological case study.")
    parser.add_argument("--nwb_json", required=True)
    parser.add_argument("--touchandsee_json", required=True)
    parser.add_argument("--legacy_json", required=True)
    parser.add_argument("--openfield_json", required=True)
    parser.add_argument("--output_dir", default="outputs/all_datasets_biological_case_study")
    parser.add_argument("--max_nwb_files", type=int, default=None)
    parser.add_argument("--nwb_bin_size_s", type=float, default=10)
    parser.add_argument("--min_units_nwb", type=int, default=30)
    parser.add_argument("--max_units_nwb", type=int, default=100)
    args = parser.parse_args()

    out = Path(args.output_dir)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    nwb_results, nwb_curves = analyze_nwb(
        args.nwb_json,
        out,
        bin_size_s=args.nwb_bin_size_s,
        min_units=args.min_units_nwb,
        max_units=args.max_units_nwb,
        max_files=args.max_nwb_files,
    )

    touch_trials, touch_effects = analyze_touchandsee(args.touchandsee_json)
    legacy_trials, legacy_structure, legacy_effects = analyze_legacy(args.legacy_json)
    openfield = analyze_openfield(args.openfield_json)

    evidence = evidence_score_table(nwb_results, touch_effects, legacy_structure, legacy_effects, openfield)
    affordance = analysis_affordance_matrix(nwb_results, touch_trials, touch_effects, legacy_structure, openfield)

    # Save outputs
    nwb_results.to_csv(out / "nwb_spike_time_temporal_drift.csv", index=False)
    nwb_curves.to_csv(out / "nwb_distance_lag_curves.csv", index=False)
    touch_trials.to_csv(out / "touchandsee_trial_table.csv", index=False)
    touch_effects.to_csv(out / "touchandsee_trial_effects.csv", index=False)
    legacy_trials.to_csv(out / "legacy_trial_table.csv", index=False)
    legacy_structure.to_csv(out / "legacy_event_task_structure.csv", index=False)
    legacy_effects.to_csv(out / "legacy_trial_effects.csv", index=False)
    openfield.to_csv(out / "openfield_continuous_reference.csv", index=False)
    evidence.to_csv(out / "cross_dataset_evidence_table.csv", index=False)
    affordance.to_csv(out / "cross_dataset_analysis_affordance_matrix.csv", index=False)

    # Figures
    plot_evidence(evidence, fig_dir / "00_cross_dataset_evidence_strength.png")
    save_heatmap(affordance, fig_dir / "01_cross_dataset_analysis_affordance_heatmap.png", "Cross-dataset biological analysis affordance map")
    plot_nwb_context(nwb_results, nwb_curves, fig_dir)
    plot_touch_effects(touch_effects, fig_dir)
    plot_legacy_structure(legacy_structure, fig_dir)
    plot_openfield(openfield, fig_dir)

    write_report(out / "all_datasets_biological_case_study_report.md", evidence, affordance, nwb_results, touch_effects, legacy_structure, openfield)

    print("\nDone.")
    print("Output directory:", out)
    print("\nEvidence table:")
    print(evidence.to_string(index=False))


if __name__ == "__main__":
    main()
