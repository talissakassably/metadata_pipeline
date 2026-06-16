# -*- coding: utf-8 -*-
r"""
Perfect cross-dataset biological case study pipeline.

Biological question
-------------------
Does the structure of experience shape neural population organization across
hippocampal-entorhinal and object-recognition electrophysiology datasets?

This script deliberately avoids the previous bad approach of forcing one noisy
proxy metric onto every dataset. Instead it does the biologically correct thing:

1. NWB hippocampal-entorhinal dataset:
   true spike-time population temporal drift analysis.
   Output: temporal organization index (TOI) by brain region and behavioral context.

2. TouchAndSee object/memory dataset:
   trial-level task modulation from extracted segment summaries.
   Output: memory vs normal, correct vs incorrect, left vs right effects on
   trial spike activity, plus trial-order drift.

3. Legacy touchscreen multisensory object dataset:
   trial/task modulation from extracted full records or previews when available.
   Output: modality/outcome/choice effects if the JSON contains enough records.

4. Openfield CA1 dataset:
   continuous spatial-foraging reference.
   Output: recording duration, sorted unit yield, LFP availability, and explicit
   statement that it is not trial-structured.

The final output is not "one fake universal boxplot"; it is a paper-style case
study showing what each dataset contributes to the same biological axis:
continuous experience vs event/trial-structured experience.

Run from the repository root
----------------------------

python case_studies\perfect_event_structure_case_study.py ^
  --nwb_json outputs\extracted_metadata\d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json ^
  --touchandsee_json outputs\extracted_metadata\p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json ^
  --legacy_json outputs\extracted_metadata\legacy_touchscreen_metadata.json ^
  --openfield_json outputs\extracted_metadata\d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json ^
  --output_dir outputs\perfect_event_structure_case_study

Optional quick test:

python case_studies\perfect_event_structure_case_study.py ^
  --nwb_json outputs\extracted_metadata\d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json ^
  --touchandsee_json outputs\extracted_metadata\p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json ^
  --legacy_json outputs\extracted_metadata\legacy_touchscreen_metadata.json ^
  --openfield_json outputs\extracted_metadata\d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json ^
  --output_dir outputs\perfect_event_structure_case_study_test ^
  --max_nwb_files 5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Generic helpers
# =============================================================================

def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, dict):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, dict):
            return default
        return int(float(x))
    except Exception:
        return default


def clean_text(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def normalize_region(x):
    s = clean_text(x).upper()

    if "LEC" in s or "LATERAL ENTORHINAL" in s:
        return "LEC"
    if "MEC" in s or "MEDIAL ENTORHINAL" in s:
        return "MEC"
    if "CA1" in s:
        return "CA1"
    if "CA2" in s:
        return "CA2"
    if "CA3" in s:
        return "CA3"
    if "DG" in s or "DENTATE" in s:
        return "DG"
    if "HIPPO" in s or "HP" in s or "HPC" in s:
        return "HPC_unspecified"
    if "PER" in s or "PRH" in s or "PERIRHINAL" in s:
        return "PER"
    if "V2" in s or "VIS" in s or "VISUAL" in s:
        return "VIS"
    if "S1" in s or "SOMATO" in s or "BARREL" in s:
        return "S1BF"
    if s.strip() == "":
        return "unknown"
    return clean_text(x)


def classify_context(description, dataset_hint=""):
    s = clean_text(description).lower()
    dataset_hint = clean_text(dataset_hint).lower()

    if "touch" in dataset_hint or "legacy" in dataset_hint:
        return "object_task"
    if "sleep" in s:
        return "sleep"
    if "sequence" in s or "seq" in s or "figure-eight" in s or "figure eight" in s:
        return "sequence_task"
    if "object" in s or "obj" in s:
        return "object_or_object_openfield"
    if "open field" in s or "foraging" in s or s.strip() == "of" or "; of" in s:
        return "open_field"
    if s.strip() == "":
        return "unknown"
    return "other"


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    out = np.empty(len(p), dtype=float)
    best = 1.0
    n = len(p)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        best = min(best, ranked[i] * n / rank)
        out[order[i]] = best
    return np.minimum(out, 1.0)


def cliffs_delta(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for a in x:
        gt += np.sum(a > y)
        lt += np.sum(a < y)
    return (gt - lt) / (len(x) * len(y))


def mannwhitney_p(x, y):
    try:
        from scipy.stats import mannwhitneyu
        x = pd.Series(x).dropna().astype(float)
        y = pd.Series(y).dropna().astype(float)
        if len(x) < 2 or len(y) < 2:
            return np.nan
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
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
# Population drift helpers for NWB
# =============================================================================

def cosine_distance_matrix(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 3:
        return None

    # Drop bad / constant columns
    keep = []
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.isfinite(col).sum() >= 5 and np.nanstd(col) > 0:
            keep.append(j)
    if len(keep) < 3:
        return None
    x = x[:, keep]

    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    norms = np.linalg.norm(z, axis=1)
    valid = norms > 0
    if valid.sum() < 5:
        return None

    z = z[valid]
    norms = norms[valid]

    sim = (z @ z.T) / np.outer(norms, norms)
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def compute_distance_lag_slope(matrix, lag_unit=10.0, max_lag=20):
    dist = cosine_distance_matrix(matrix)
    if dist is None:
        return np.nan, pd.DataFrame(), None

    n = dist.shape[0]
    max_lag = min(max_lag, n - 1)
    rows = []
    for lag in range(1, max_lag + 1):
        vals = np.diag(dist, k=lag)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
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


def bin_spike_times(spike_times_by_unit, bin_size_s=10.0, max_units=100, min_units=30):
    cleaned = []
    for st in spike_times_by_unit:
        arr = np.asarray(st, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        arr = arr[arr >= 0]
        if arr.size > 2:
            cleaned.append(arr)

    if len(cleaned) < min_units:
        return None, None

    if len(cleaned) > max_units:
        rng = random.Random(42)
        cleaned = rng.sample(cleaned, max_units)

    t_min = min(float(np.min(x)) for x in cleaned)
    t_max = max(float(np.max(x)) for x in cleaned)

    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return None, None

    cleaned = [x - t_min for x in cleaned]
    duration = t_max - t_min
    if duration < bin_size_s * 8:
        return None, None

    edges = np.arange(0, duration + bin_size_s, bin_size_s)
    if len(edges) < 8:
        return None, None

    matrix = np.zeros((len(edges) - 1, len(cleaned)), dtype=float)
    for j, st in enumerate(cleaned):
        matrix[:, j] = np.histogram(st, bins=edges)[0]

    return matrix, edges


# =============================================================================
# NWB true spike-time population drift
# =============================================================================

def nwb_file_metadata(nwb_json):
    data = load_json(nwb_json)
    rows = []
    for item in data.get("files", []):
        meta = item.get("file_metadata", {}) or {}
        nwb = item.get("nwb_extraction", {}) or {}
        if not nwb.get("success"):
            continue
        region_text = ";".join(clean_text(x) for x in (nwb.get("electrode_locations", []) or []))
        rows.append({
            "absolute_path": meta.get("absolute_path"),
            "relative_path": meta.get("path"),
            "session_id": nwb.get("identifier") or meta.get("file_name"),
            "subject_id": (nwb.get("subject") or {}).get("subject_id"),
            "description": nwb.get("session_description"),
            "context": classify_context(nwb.get("session_description"), "nwb"),
            "file_region_text": region_text,
            "file_region": normalize_region(region_text),
            "n_units_metadata": safe_int(nwb.get("n_units")),
            "n_electrodes": safe_int(nwb.get("n_electrodes")),
        })
    return pd.DataFrame(rows)


def infer_unit_regions(nwbfile, units_df, fallback_region):
    """
    Try to infer one region per unit from NWB tables.

    If unit-level region mapping fails, falls back to the file-level region.
    """
    n = len(units_df)
    regions = [fallback_region] * n

    # Direct columns first
    for col in ["location", "brain_region", "region", "structure"]:
        if col in units_df.columns:
            vals = [normalize_region(v) for v in units_df[col].values]
            if len(set(vals)) > 1 or vals[0] != "unknown":
                return vals

    # Electrode-table mapping if possible
    try:
        electrodes_df = nwbfile.electrodes.to_dataframe()
        location_col = None
        for col in ["location", "brain_region", "structure"]:
            if col in electrodes_df.columns:
                location_col = col
                break

        if location_col is not None and "electrodes" in units_df.columns:
            mapped = []
            for _, row in units_df.iterrows():
                er = row["electrodes"]
                locs = []

                # er may be a DynamicTableRegion, list, ndarray, slice, or DataFrame-like
                try:
                    if hasattr(er, "to_dataframe"):
                        edf = er.to_dataframe()
                        if location_col in edf.columns:
                            locs = list(edf[location_col].values)
                    elif isinstance(er, (list, tuple, np.ndarray, pd.Series)):
                        for idx in list(er):
                            try:
                                if idx in electrodes_df.index:
                                    locs.append(electrodes_df.loc[idx, location_col])
                                else:
                                    locs.append(electrodes_df.iloc[int(idx)][location_col])
                            except Exception:
                                pass
                except Exception:
                    pass

                if locs:
                    norm = [normalize_region(x) for x in locs]
                    # majority
                    mapped.append(pd.Series(norm).mode().iloc[0])
                else:
                    mapped.append(fallback_region)

            if mapped:
                return mapped
    except Exception:
        pass

    return regions


def read_nwb_units(path):
    try:
        from pynwb import NWBHDF5IO
    except Exception as e:
        raise RuntimeError("pynwb is required. Install with: python -m pip install pynwb") from e

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        if "spike_times" not in units_df.columns:
            return [], units_df, None
        spike_times = [np.asarray(st, dtype=float) for st in units_df["spike_times"].values]
        return spike_times, units_df, nwbfile


def run_nwb_analysis(nwb_json, out_dir, bin_size_s=10.0, min_units=30, max_units=100, max_files=None):
    meta = nwb_file_metadata(nwb_json)
    results = []
    curves = []
    example_saved = 0

    for i, row in meta.iterrows():
        if max_files is not None and i >= max_files:
            break

        path = Path(clean_text(row["absolute_path"]))
        if not path.exists():
            results.append({
                "dataset": "NWB hippocampal-entorhinal",
                "session_id": row["session_id"],
                "subject_id": row["subject_id"],
                "brain_region": row["file_region"],
                "behavioral_context": row["context"],
                "analysis": "NWB true spike-time temporal drift",
                "metric": "TOI",
                "value": np.nan,
                "n_units": 0,
                "n_bins": 0,
                "status": "missing_file",
                "evidence_level": "raw_spike_times",
            })
            continue

        print(f"NWB {i+1}/{len(meta)}: {row['session_id']} | {row['context']}")

        try:
            spike_times, units_df, nwbfile = read_nwb_units(path)
            unit_regions = infer_unit_regions(nwbfile, units_df, row["file_region"])

            # group by region, but also keep a fallback all-LEC if only file-level LEC.
            reg_to_spikes = {}
            for st, reg in zip(spike_times, unit_regions):
                reg = normalize_region(reg)
                reg_to_spikes.setdefault(reg, []).append(st)

            for region, sts in sorted(reg_to_spikes.items()):
                if region == "unknown":
                    continue

                matrix, edges = bin_spike_times(
                    sts,
                    bin_size_s=bin_size_s,
                    max_units=max_units,
                    min_units=min_units,
                )

                if matrix is None:
                    results.append({
                        "dataset": "NWB hippocampal-entorhinal",
                        "session_id": row["session_id"],
                        "subject_id": row["subject_id"],
                        "brain_region": region,
                        "behavioral_context": row["context"],
                        "analysis": "NWB true spike-time temporal drift",
                        "metric": "TOI",
                        "value": np.nan,
                        "n_units": len(sts),
                        "n_bins": 0,
                        "status": "insufficient_units_or_duration",
                        "evidence_level": "raw_spike_times",
                    })
                    continue

                toi, curve, dist = compute_distance_lag_slope(matrix, lag_unit=bin_size_s, max_lag=20)

                results.append({
                    "dataset": "NWB hippocampal-entorhinal",
                    "session_id": row["session_id"],
                    "subject_id": row["subject_id"],
                    "brain_region": region,
                    "behavioral_context": row["context"],
                    "analysis": "NWB true spike-time temporal drift",
                    "metric": "TOI",
                    "value": toi,
                    "n_units": matrix.shape[1],
                    "n_bins": matrix.shape[0],
                    "status": "ok" if np.isfinite(toi) else "toi_failed",
                    "evidence_level": "raw_spike_times",
                })

                if not curve.empty:
                    curve["dataset"] = "NWB hippocampal-entorhinal"
                    curve["session_id"] = row["session_id"]
                    curve["brain_region"] = region
                    curve["behavioral_context"] = row["context"]
                    curves.append(curve)

                if dist is not None and example_saved < 3:
                    plt.figure(figsize=(6, 5))
                    plt.imshow(dist, aspect="auto", origin="lower")
                    plt.colorbar(label="Cosine distance")
                    plt.xlabel("Time bin")
                    plt.ylabel("Time bin")
                    plt.title(f"NWB {region} {row['context']}")
                    plt.tight_layout()
                    plt.savefig(out_dir / "figures" / f"example_nwb_distance_{example_saved+1}_{region}_{row['context']}.png", dpi=250)
                    plt.close()
                    example_saved += 1

        except Exception as e:
            results.append({
                "dataset": "NWB hippocampal-entorhinal",
                "session_id": row["session_id"],
                "subject_id": row["subject_id"],
                "brain_region": row["file_region"],
                "behavioral_context": row["context"],
                "analysis": "NWB true spike-time temporal drift",
                "metric": "TOI",
                "value": np.nan,
                "n_units": row["n_units_metadata"],
                "n_bins": 0,
                "status": "error: " + repr(e),
                "evidence_level": "raw_spike_times",
            })

    return (
        pd.DataFrame(results),
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
    )


# =============================================================================
# TouchAndSee trial-level task modulation from JSON
# =============================================================================

def get_segments(obj):
    segments = obj.get("segments", [])
    if isinstance(segments, dict):
        if "preview" in segments:
            return segments.get("preview") or [], "preview"
        return [], "none"
    if isinstance(segments, list):
        return segments, "full"
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

        segments, level = get_segments(obj)
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            ann = seg.get("annotation_preview", {}) or {}
            rows.append({
                "dataset": "TouchAndSee object-memory",
                "session_id": meta.get("session_id") or meta.get("file_name"),
                "subject_id": meta.get("subject_id") or meta.get("animal_id"),
                "trial_number": safe_float(ann.get("trial_number")),
                "trial_outcome": clean_text(ann.get("trial_outcome")),
                "trial_side": clean_text(ann.get("trial_side")),
                "trial_type": clean_text(ann.get("trial_type")),
                "n_spikes_total": safe_float(seg.get("n_spikes_total")),
                "n_spiketrains": safe_float(seg.get("n_spiketrains")),
                "mean_spikes_per_spiketrain": safe_float(seg.get("mean_spikes_per_spiketrain")),
                "n_events": safe_float(seg.get("n_events")),
                "n_event_times_total": safe_float(seg.get("n_event_times_total")),
                "record_level": level,
            })

    return pd.DataFrame(rows)


def analyze_touchandsee(touch_json):
    trials = touchandsee_trial_table(touch_json)
    effects = []

    if trials.empty:
        return trials, pd.DataFrame(effects)

    # Use mean spikes per spiketrain when available; otherwise total spikes
    trials["activity"] = pd.to_numeric(trials["mean_spikes_per_spiketrain"], errors="coerce")
    missing = trials["activity"].isna()
    trials.loc[missing, "activity"] = pd.to_numeric(trials.loc[missing, "n_spikes_total"], errors="coerce")

    for session_id, sub in trials.groupby("session_id"):
        subject = sub["subject_id"].iloc[0] if "subject_id" in sub else ""
        n_trials = len(sub)

        # Memory vs normal
        if sub["trial_type"].astype(str).str.len().gt(0).any():
            normal = sub.loc[sub["trial_type"].str.lower() == "normal", "activity"]
            memory = sub.loc[sub["trial_type"].str.lower() == "memory", "activity"]
            effects.append({
                "dataset": "TouchAndSee object-memory",
                "session_id": session_id,
                "subject_id": subject,
                "analysis": "memory_vs_normal_trial_activity",
                "contrast": "memory - normal",
                "effect_size_cliffs_delta": cliffs_delta(memory, normal),
                "p_value": mannwhitney_p(memory, normal),
                "n_group_a": int(memory.dropna().shape[0]),
                "n_group_b": int(normal.dropna().shape[0]),
                "n_trials": n_trials,
                "status": "ok" if len(memory.dropna()) >= 2 and len(normal.dropna()) >= 2 else "insufficient_groups",
                "evidence_level": "trial_segment_summary",
            })

        # Correct vs incorrect
        if sub["trial_outcome"].astype(str).str.len().gt(0).any():
            correct = sub.loc[sub["trial_outcome"].str.lower() == "correct", "activity"]
            incorrect = sub.loc[sub["trial_outcome"].str.lower() == "incorrect", "activity"]
            effects.append({
                "dataset": "TouchAndSee object-memory",
                "session_id": session_id,
                "subject_id": subject,
                "analysis": "correct_vs_incorrect_trial_activity",
                "contrast": "correct - incorrect",
                "effect_size_cliffs_delta": cliffs_delta(correct, incorrect),
                "p_value": mannwhitney_p(correct, incorrect),
                "n_group_a": int(correct.dropna().shape[0]),
                "n_group_b": int(incorrect.dropna().shape[0]),
                "n_trials": n_trials,
                "status": "ok" if len(correct.dropna()) >= 2 and len(incorrect.dropna()) >= 2 else "insufficient_groups",
                "evidence_level": "trial_segment_summary",
            })

        # Trial-order drift
        r, p = spearman_r(sub["trial_number"], sub["activity"])
        effects.append({
            "dataset": "TouchAndSee object-memory",
            "session_id": session_id,
            "subject_id": subject,
            "analysis": "trial_order_activity_drift",
            "contrast": "activity ~ trial_number",
            "effect_size_cliffs_delta": np.nan,
            "spearman_r": r,
            "p_value": p,
            "n_group_a": np.nan,
            "n_group_b": np.nan,
            "n_trials": n_trials,
            "status": "ok" if np.isfinite(r) else "insufficient_trial_order",
            "evidence_level": "trial_segment_summary",
        })

    effects = pd.DataFrame(effects)
    if not effects.empty:
        mask = effects["p_value"].notna()
        effects.loc[mask, "p_value_bh"] = bh_adjust(effects.loc[mask, "p_value"].values)
    return trials, effects


# =============================================================================
# Legacy touchscreen trial-level effects from JSON
# =============================================================================

def find_records(session, candidates):
    for key in candidates:
        value = session.get(key)
        if isinstance(value, list) and len(value) > 0:
            return value, key
    return [], ""


def legacy_trial_table(legacy_json):
    data = load_json(legacy_json)
    rows = []
    failures = []

    for sess in data.get("sessions", []):
        sid = sess.get("session_id")
        subject = sess.get("animal_id") or sess.get("subject_id")
        summary = sess.get("summary", {}) or {}
        records, source_key = find_records(
            sess,
            [
                "trial_records",
                "trial_dataframe_records",
                "trial_table_records",
                "trials_records",
                "trial_preview_records",
            ],
        )

        if not records:
            failures.append({
                "session_id": sid,
                "subject_id": subject,
                "reason": "no_trial_records_in_json",
                "n_trials_summary": safe_int(summary.get("n_trials_from_dataframe")),
            })
            continue

        for rec in records:
            if not isinstance(rec, dict):
                continue
            rows.append({
                "dataset": "Legacy touchscreen multisensory object",
                "session_id": sid,
                "subject_id": subject,
                "source_key": source_key,
                "trial_id": rec.get("trial_id") or rec.get("trial_number"),
                "modality": clean_text(rec.get("modality")),
                "object": clean_text(rec.get("object")),
                "choice": clean_text(rec.get("choice")),
                "correct": clean_text(rec.get("correct")),
                "trial_outcome": clean_text(rec.get("trial_outcome")),
                "trial_side": clean_text(rec.get("trial_side")),
                "trial_type": clean_text(rec.get("trial_type")),
                "activity": safe_float(
                    rec.get("n_spikes_total",
                        rec.get("n_spikes",
                            rec.get("spike_count",
                                rec.get("num_spikes", np.nan))))
                ),
            })

    return pd.DataFrame(rows), pd.DataFrame(failures)


def analyze_legacy(legacy_json):
    trials, failures = legacy_trial_table(legacy_json)
    effects = []

    if trials.empty:
        return trials, effects_df_with_failures(pd.DataFrame(), failures)

    for sid, sub in trials.groupby("session_id"):
        subject = sub["subject_id"].iloc[0] if "subject_id" in sub else ""
        n_trials = len(sub)

        # If activity is absent, still quantify task variable availability
        has_activity = pd.to_numeric(sub["activity"], errors="coerce").notna().sum() >= 4

        if has_activity:
            activity = pd.to_numeric(sub["activity"], errors="coerce")
            # Correct vs incorrect
            if sub["correct"].astype(str).str.len().gt(0).any():
                a = activity[sub["correct"].astype(str).str.lower().isin(["1", "true", "correct"])]
                b = activity[sub["correct"].astype(str).str.lower().isin(["0", "false", "incorrect"])]
                effects.append({
                    "dataset": "Legacy touchscreen multisensory object",
                    "session_id": sid,
                    "subject_id": subject,
                    "analysis": "correct_vs_incorrect_trial_activity",
                    "contrast": "correct - incorrect",
                    "effect_size_cliffs_delta": cliffs_delta(a, b),
                    "p_value": mannwhitney_p(a, b),
                    "n_group_a": int(a.dropna().shape[0]),
                    "n_group_b": int(b.dropna().shape[0]),
                    "n_trials": n_trials,
                    "status": "ok" if len(a.dropna()) >= 2 and len(b.dropna()) >= 2 else "insufficient_groups",
                    "evidence_level": "legacy_trial_records",
                })

            # Modality differences: do pairwise first two most common modalities
            if sub["modality"].astype(str).str.len().gt(0).any():
                counts = sub["modality"].value_counts()
                if len(counts) >= 2:
                    m1, m2 = counts.index[:2]
                    a = activity[sub["modality"] == m1]
                    b = activity[sub["modality"] == m2]
                    effects.append({
                        "dataset": "Legacy touchscreen multisensory object",
                        "session_id": sid,
                        "subject_id": subject,
                        "analysis": "modality_trial_activity",
                        "contrast": f"{m1} - {m2}",
                        "effect_size_cliffs_delta": cliffs_delta(a, b),
                        "p_value": mannwhitney_p(a, b),
                        "n_group_a": int(a.dropna().shape[0]),
                        "n_group_b": int(b.dropna().shape[0]),
                        "n_trials": n_trials,
                        "status": "ok" if len(a.dropna()) >= 2 and len(b.dropna()) >= 2 else "insufficient_groups",
                        "evidence_level": "legacy_trial_records",
                    })
        else:
            # Important: still record that the task design is extractable, but
            # no activity per trial is available from JSON.
            effects.append({
                "dataset": "Legacy touchscreen multisensory object",
                "session_id": sid,
                "subject_id": subject,
                "analysis": "trial_task_structure_available",
                "contrast": "task variables present, no trial activity",
                "effect_size_cliffs_delta": np.nan,
                "p_value": np.nan,
                "n_group_a": np.nan,
                "n_group_b": np.nan,
                "n_trials": n_trials,
                "status": "task_structure_only",
                "evidence_level": "legacy_trial_records_no_activity",
            })

    effects = pd.DataFrame(effects)
    if not effects.empty:
        mask = effects["p_value"].notna()
        effects.loc[mask, "p_value_bh"] = bh_adjust(effects.loc[mask, "p_value"].values)
    return trials, effects_df_with_failures(effects, failures)


def effects_df_with_failures(effects, failures):
    if failures is None or failures.empty:
        return effects
    fail_effects = []
    for _, row in failures.iterrows():
        fail_effects.append({
            "dataset": "Legacy touchscreen multisensory object",
            "session_id": row.get("session_id"),
            "subject_id": row.get("subject_id"),
            "analysis": "legacy_json_trial_extraction",
            "contrast": row.get("reason"),
            "effect_size_cliffs_delta": np.nan,
            "p_value": np.nan,
            "n_group_a": np.nan,
            "n_group_b": np.nan,
            "n_trials": row.get("n_trials_summary"),
            "status": "not_usable_for_trial_effects",
            "evidence_level": "json_summary_only",
        })
    return pd.concat([effects, pd.DataFrame(fail_effects)], ignore_index=True, sort=False)


# =============================================================================
# Openfield continuous reference
# =============================================================================

def summarize_openfield(openfield_json):
    data = load_json(openfield_json)
    rows = []

    for sess in data.get("sessions", []):
        summary = sess.get("summary", {}) or {}
        neuralynx = sess.get("neuralynx_metadata", {}) or {}
        mclust = sess.get("mclust_metadata", {}) or {}
        rows.append({
            "dataset": "Openfield CA1 continuous spatial foraging",
            "session_id": sess.get("session_id"),
            "subject_id": sess.get("subject_id"),
            "recording_system": sess.get("recording_system"),
            "recording_duration_s": safe_float(summary.get("recording_duration_s")),
            "sorted_units_mclust": safe_int(summary.get("sorted_units_mclust", mclust.get("sorted_units_mclust"))),
            "sorted_unit_spikes_total_mclust": safe_int(summary.get("sorted_unit_spikes_total_mclust", mclust.get("sorted_unit_spikes_total_mclust"))),
            "raw_spike_events_total": safe_int(summary.get("raw_spike_events_total", neuralynx.get("raw_spike_events_total"))),
            "n_lfp_channels": safe_int(summary.get("n_lfp_channels", neuralynx.get("n_lfp_channels"))),
            "has_position_metadata": bool(summary.get("has_position_metadata")),
            "has_sorted_unit_metadata": bool(summary.get("has_sorted_unit_metadata")),
            "biological_role": "continuous_spatial_reference",
        })

    return pd.DataFrame(rows)


# =============================================================================
# Synthesis and figures
# =============================================================================

def make_evidence_table(nwb_results, touch_effects, legacy_effects, openfield_summary):
    rows = []

    # NWB
    ok_nwb = nwb_results[nwb_results["status"].eq("ok")] if not nwb_results.empty else pd.DataFrame()
    rows.append({
        "dataset": "NWB hippocampal-entorhinal",
        "original_biological_context": "LEC/MEC/CA1 event structure, sequence, object, sleep, open field",
        "case_study_contribution": "Tests true spike-time population temporal drift by region and context",
        "main_metric": "TOI from binned spike-time population vectors",
        "usable_units": int(ok_nwb["session_id"].nunique()) if not ok_nwb.empty else 0,
        "evidence_strength": "strong" if len(ok_nwb) >= 10 else "limited",
    })

    # TouchAndSee
    ok_touch = touch_effects[touch_effects["status"].eq("ok")] if not touch_effects.empty else pd.DataFrame()
    rows.append({
        "dataset": "TouchAndSee object-memory",
        "original_biological_context": "trial-based object/memory behavior",
        "case_study_contribution": "Tests whether trial type/outcome is reflected in trial activity summaries",
        "main_metric": "Cliff's delta for memory vs normal, correct vs incorrect, and trial-order drift",
        "usable_units": int(ok_touch["session_id"].nunique()) if not ok_touch.empty else 0,
        "evidence_strength": "moderate" if len(ok_touch) >= 10 else ("limited" if len(ok_touch) > 0 else "not usable"),
    })

    # Legacy
    ok_leg = legacy_effects[legacy_effects["status"].eq("ok")] if not legacy_effects.empty else pd.DataFrame()
    rows.append({
        "dataset": "Legacy touchscreen multisensory object",
        "original_biological_context": "visual/tactile/multisensory object and outcome task",
        "case_study_contribution": "Tests whether extracted trial records support outcome/modality activity analysis",
        "main_metric": "Trial-level contrast effects if activity records are present",
        "usable_units": int(ok_leg["session_id"].nunique()) if not ok_leg.empty else 0,
        "evidence_strength": "moderate" if len(ok_leg) >= 5 else ("limited" if len(ok_leg) > 0 else "task metadata only / not usable for activity"),
    })

    # Openfield
    ok_open = openfield_summary[openfield_summary["has_sorted_unit_metadata"].eq(True)] if not openfield_summary.empty else pd.DataFrame()
    rows.append({
        "dataset": "Openfield CA1 continuous spatial foraging",
        "original_biological_context": "continuous open-field CA1 spatial foraging",
        "case_study_contribution": "Provides continuous spatial reference and contrasts absence of trial/event structure",
        "main_metric": "recording duration, sorted unit yield, LFP availability",
        "usable_units": int(ok_open["session_id"].nunique()) if not ok_open.empty else 0,
        "evidence_strength": "contextual_reference",
    })

    return pd.DataFrame(rows)


def save_barh(values, labels, title, xlabel, path):
    if len(values) == 0:
        return
    order = np.argsort(values)
    values = np.asarray(values)[order]
    labels = np.asarray(labels)[order]

    plt.figure(figsize=(9, max(4, len(values) * 0.45)))
    plt.barh(range(len(values)), values)
    plt.yticks(range(len(values)), labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def plot_nwb(nwb_results, curves, fig_dir):
    ok = nwb_results[nwb_results["status"].eq("ok")].copy()
    ok["value"] = pd.to_numeric(ok["value"], errors="coerce")
    ok = ok.dropna(subset=["value"])

    if ok.empty:
        return

    # TOI by context
    groups = []
    labels = []
    for ctx in sorted(ok["behavioral_context"].dropna().unique()):
        vals = ok.loc[ok["behavioral_context"] == ctx, "value"].dropna().values
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
        plt.savefig(fig_dir / "01_nwb_toi_by_context.png", dpi=250)
        plt.close()

    # TOI by region
    groups = []
    labels = []
    for reg in ["LEC", "MEC", "CA1", "CA3", "HPC_unspecified"]:
        vals = ok.loc[ok["brain_region"] == reg, "value"].dropna().values
        if len(vals) >= 2:
            groups.append(vals)
            labels.append(reg)
    if groups:
        plt.figure(figsize=(8, 5))
        try:
            plt.boxplot(groups, tick_labels=labels, showfliers=False)
        except TypeError:
            plt.boxplot(groups, labels=labels, showfliers=False)
        plt.ylabel("TOI: distance-lag slope")
        plt.title("NWB spike-time temporal organization by brain region")
        plt.tight_layout()
        plt.savefig(fig_dir / "02_nwb_toi_by_region.png", dpi=250)
        plt.close()

    # Mean lag curves
    if not curves.empty:
        plt.figure(figsize=(8, 5))
        for ctx, sub in curves.groupby("behavioral_context"):
            mean_curve = sub.groupby("lag_value")["mean_distance"].mean().reset_index()
            plt.plot(mean_curve["lag_value"], mean_curve["mean_distance"], marker="o", label=ctx)
        plt.xlabel("Time lag (s)")
        plt.ylabel("Mean population distance")
        plt.title("NWB distance-lag curves by behavioral context")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "03_nwb_distance_lag_by_context.png", dpi=250)
        plt.close()


def plot_touch(touch_effects, fig_dir):
    if touch_effects.empty:
        return
    ok = touch_effects[touch_effects["status"].eq("ok")].copy()
    if ok.empty:
        return

    for analysis in ["memory_vs_normal_trial_activity", "correct_vs_incorrect_trial_activity", "trial_order_activity_drift"]:
        sub = ok[ok["analysis"].eq(analysis)]
        if sub.empty:
            continue
        if analysis == "trial_order_activity_drift":
            values = pd.to_numeric(sub["spearman_r"], errors="coerce").dropna()
            title = "TouchAndSee trial-order activity drift"
            xlabel = "Spearman r(activity, trial number)"
        else:
            values = pd.to_numeric(sub["effect_size_cliffs_delta"], errors="coerce").dropna()
            title = "TouchAndSee " + analysis.replace("_", " ")
            xlabel = "Cliff's delta"

        if len(values) == 0:
            continue
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=15)
        plt.axvline(0, linestyle="--")
        plt.xlabel(xlabel)
        plt.ylabel("Sessions")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_dir / f"04_touchandsee_{analysis}.png", dpi=250)
        plt.close()


def plot_openfield(openfield_summary, fig_dir):
    if openfield_summary.empty:
        return
    df = openfield_summary.copy()

    if "sorted_units_mclust" in df:
        plt.figure(figsize=(8, 4))
        plt.hist(pd.to_numeric(df["sorted_units_mclust"], errors="coerce").dropna(), bins=20)
        plt.xlabel("MClust sorted units per session")
        plt.ylabel("Sessions")
        plt.title("Openfield CA1 sorted unit yield")
        plt.tight_layout()
        plt.savefig(fig_dir / "05_openfield_sorted_unit_yield.png", dpi=250)
        plt.close()

    if "recording_duration_s" in df:
        plt.figure(figsize=(8, 4))
        vals = pd.to_numeric(df["recording_duration_s"], errors="coerce").dropna() / 60
        plt.hist(vals, bins=20)
        plt.xlabel("Recording duration (min)")
        plt.ylabel("Sessions")
        plt.title("Openfield CA1 continuous recording duration")
        plt.tight_layout()
        plt.savefig(fig_dir / "06_openfield_recording_duration.png", dpi=250)
        plt.close()


def plot_evidence(evidence, fig_dir):
    # ordinal evidence strength
    mapping = {
        "not usable": 0,
        "task metadata only / not usable for activity": 1,
        "limited": 2,
        "contextual_reference": 3,
        "moderate": 4,
        "strong": 5,
    }
    values = [mapping.get(x, 0) for x in evidence["evidence_strength"]]
    labels = evidence["dataset"].tolist()
    save_barh(
        values,
        labels,
        "Evidence strength by dataset for the case study",
        "Evidence strength score",
        fig_dir / "00_case_study_evidence_strength.png",
    )


def write_report(out_path, evidence, nwb_results, touch_effects, legacy_effects, openfield_summary):
    ok_nwb = nwb_results[nwb_results["status"].eq("ok")] if not nwb_results.empty else pd.DataFrame()
    ok_touch = touch_effects[touch_effects["status"].eq("ok")] if not touch_effects.empty else pd.DataFrame()
    ok_legacy = legacy_effects[legacy_effects["status"].eq("ok")] if not legacy_effects.empty else pd.DataFrame()

    lines = [
        "# Perfect event-structure cross-dataset case study",
        "",
        "## Biological question",
        "",
        "Does the structure of experience shape neural population organization across hippocampal-entorhinal and object-recognition electrophysiology datasets?",
        "",
        "## Core idea",
        "",
        "The datasets are not forced into a single artificial metric. Instead, each dataset contributes the biologically valid analysis supported by its structure:",
        "",
        "- NWB hippocampal-entorhinal recordings: true spike-time population temporal drift.",
        "- TouchAndSee object-memory recordings: trial-level activity modulation by memory demand and outcome.",
        "- Legacy touchscreen multisensory recordings: trial/task modulation when full records are available.",
        "- Openfield CA1 recordings: continuous spatial-foraging reference, sorted unit yield, duration and LFP availability.",
        "",
        "## Evidence table",
        "",
    ]

    for _, row in evidence.iterrows():
        lines.append(f"### {row['dataset']}")
        lines.append(f"- Biological context: {row['original_biological_context']}")
        lines.append(f"- Contribution: {row['case_study_contribution']}")
        lines.append(f"- Metric: {row['main_metric']}")
        lines.append(f"- Usable sessions/rows: {row['usable_units']}")
        lines.append(f"- Evidence strength: {row['evidence_strength']}")
        lines.append("")

    lines += [
        "## Main results summary",
        "",
        f"- NWB usable region/session analyses: {len(ok_nwb)} rows across {ok_nwb['session_id'].nunique() if not ok_nwb.empty else 0} sessions.",
        f"- TouchAndSee usable trial-effect analyses: {len(ok_touch)} rows across {ok_touch['session_id'].nunique() if not ok_touch.empty else 0} sessions.",
        f"- Legacy usable trial-effect analyses: {len(ok_legacy)} rows across {ok_legacy['session_id'].nunique() if not ok_legacy.empty else 0} sessions.",
        f"- Openfield reference sessions: {len(openfield_summary)} sessions.",
        "",
        "## Recommended interpretation",
        "",
        "This case study contributes biologically by showing how the structure of experience determines what can be measured from reused electrophysiology datasets. NWB supports population temporal drift analysis across hippocampal-entorhinal regions and behavioral contexts. TouchAndSee supports object/memory trial-level modulation. Legacy touchscreen supports multisensory object-task reuse when trial records are available. Openfield CA1 provides the continuous spatial reference condition.",
        "",
        "## What not to claim",
        "",
        "Do not claim that all datasets were analyzed with identical raw spike-time methods. The correct claim is that a single biological axis — continuous versus event/trial-structured experience — was tested using the strongest valid representation available for each dataset.",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run perfect cross-dataset biological case study.")
    parser.add_argument("--nwb_json", required=True)
    parser.add_argument("--touchandsee_json", required=True)
    parser.add_argument("--legacy_json", required=True)
    parser.add_argument("--openfield_json", required=True)
    parser.add_argument("--output_dir", default="outputs/perfect_event_structure_case_study")
    parser.add_argument("--nwb_bin_size_s", type=float, default=10.0)
    parser.add_argument("--min_units_nwb", type=int, default=30)
    parser.add_argument("--max_units_nwb", type=int, default=100)
    parser.add_argument("--max_nwb_files", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. NWB true spike-time drift
    nwb_results, nwb_curves = run_nwb_analysis(
        args.nwb_json,
        out,
        bin_size_s=args.nwb_bin_size_s,
        min_units=args.min_units_nwb,
        max_units=args.max_units_nwb,
        max_files=args.max_nwb_files,
    )

    # 2. TouchAndSee trial-level effects
    touch_trials, touch_effects = analyze_touchandsee(args.touchandsee_json)

    # 3. Legacy touchscreen trial-level effects
    legacy_trials, legacy_effects = analyze_legacy(args.legacy_json)

    # 4. Openfield continuous reference
    openfield_summary = summarize_openfield(args.openfield_json)

    # 5. Evidence table and report
    evidence = make_evidence_table(nwb_results, touch_effects, legacy_effects, openfield_summary)

    # Save all data
    nwb_results.to_csv(out / "nwb_temporal_drift_results.csv", index=False)
    nwb_curves.to_csv(out / "nwb_distance_lag_curves.csv", index=False)
    touch_trials.to_csv(out / "touchandsee_trial_table.csv", index=False)
    touch_effects.to_csv(out / "touchandsee_trial_effects.csv", index=False)
    legacy_trials.to_csv(out / "legacy_trial_table.csv", index=False)
    legacy_effects.to_csv(out / "legacy_trial_effects.csv", index=False)
    openfield_summary.to_csv(out / "openfield_continuous_reference.csv", index=False)
    evidence.to_csv(out / "cross_dataset_evidence_table.csv", index=False)

    # Figures
    plot_evidence(evidence, fig_dir)
    plot_nwb(nwb_results, nwb_curves, fig_dir)
    plot_touch(touch_effects, fig_dir)
    plot_openfield(openfield_summary, fig_dir)

    write_report(out / "perfect_event_structure_case_study_report.md", evidence, nwb_results, touch_effects, legacy_effects, openfield_summary)

    print("\nDone.")
    print("Output directory:", out)
    print("\nEvidence table:")
    print(evidence.to_string(index=False))


if __name__ == "__main__":
    main()
