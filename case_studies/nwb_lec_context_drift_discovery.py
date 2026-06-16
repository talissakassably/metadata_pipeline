# -*- coding: utf-8 -*-
r"""
Second-layer discovery pipeline:
Context-dependent temporal drift in LEC population activity.

Purpose
-------
This is the "paper-style neuroscience" layer that sits on top of the
cross-dataset affordance/reuse pipeline.

It focuses on the strongest raw neural dataset:
NWB hippocampal-entorhinal recordings.

Biological question
-------------------
Does behavioral/event context modulate temporal drift in LEC population activity?

Hypothesis
----------
LEC population states should diverge more strongly over time in structured
experience contexts such as sequence/object sessions than in simple open-field
sessions.

What it does
------------
1. Loads NWB files from the extracted NWB metadata JSON.
2. Keeps sessions whose region metadata is LEC.
3. Recodes messy session descriptions into clean contexts:
   open_field, object_context, sequence_task, sleep.
4. Extracts unit spike times.
5. Computes population vectors in time bins.
6. Computes temporal organization index (TOI):
      slope(mean population distance ~ time lag)
7. Runs controls:
   - fixed unit-count subsampling
   - repeated bootstrap/subsampling
   - optional duration-matched window
8. Runs statistics:
   - Kruskal-Wallis across contexts
   - pairwise Mann-Whitney tests
   - Cliff's delta effect size
   - Spearman correlation with number of units and duration
9. Generates publication-style figures.

Run
---
python case_studies\nwb_lec_context_drift_discovery.py ^
  --nwb_json outputs\extracted_metadata\d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json ^
  --output_dir outputs\nwb_lec_context_drift_discovery

Quick test:
add --max_files 8 --n_repeats 5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Helpers
# =============================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(x, default=0):
    try:
        if x is None or isinstance(x, dict):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=np.nan):
    try:
        if x is None or isinstance(x, dict):
            return default
        return float(x)
    except Exception:
        return default


def txt(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def normalize_region(x):
    s = txt(x).upper()
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
    return txt(x) if txt(x).strip() else "unknown"


def clean_context(description):
    """
    Recode messy session descriptions from the NWB metadata into analysis contexts.

    This is intentionally conservative.
    """
    s = txt(description).lower()

    # remove punctuation variants
    s2 = re.sub(r"[^a-z0-9]+", " ", s)

    if "sleep" in s2:
        return "sleep"

    if "sequence" in s2 or re.search(r"\bseq\b", s2):
        return "sequence_task"

    # Object with open field, or pure object, is treated as object/event context.
    if "object" in s2 or re.search(r"\bobj\b", s2):
        return "object_context"

    # OF / open field
    if "open field" in s or re.search(r"\bof\b", s2) or "foraging" in s2:
        return "open_field"

    return "unclear"


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


def kruskal_test(groups_dict):
    try:
        from scipy.stats import kruskal
        groups = []
        names = []
        for name, vals in groups_dict.items():
            vals = pd.Series(vals).dropna().astype(float).values
            if len(vals) >= 2:
                groups.append(vals)
                names.append(name)
        if len(groups) < 2:
            return np.nan, np.nan, names
        h, p = kruskal(*groups)
        return float(h), float(p), names
    except Exception:
        return np.nan, np.nan, []


def spearman_test(x, y):
    try:
        from scipy.stats import spearmanr
        x = pd.Series(x)
        y = pd.Series(y)
        ok = x.notna() & y.notna()
        if ok.sum() < 4:
            return np.nan, np.nan
        r, p = spearmanr(x[ok].astype(float), y[ok].astype(float))
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


# =============================================================================
# NWB loading
# =============================================================================

def extract_nwb_metadata_table(nwb_json):
    data = load_json(nwb_json)
    rows = []

    for item in data.get("files", []):
        meta = item.get("file_metadata", {}) or {}
        nwb = item.get("nwb_extraction", {}) or {}
        if not nwb.get("success"):
            continue

        regions = nwb.get("electrode_locations", []) or []
        region_text = ";".join(txt(r) for r in regions)
        region = normalize_region(region_text)
        desc = nwb.get("session_description")

        rows.append({
            "absolute_path": meta.get("absolute_path"),
            "file_name": meta.get("file_name"),
            "session_id": nwb.get("identifier") or meta.get("file_name"),
            "subject_id": (nwb.get("subject") or {}).get("subject_id"),
            "description": desc,
            "clean_context": clean_context(desc),
            "region_text": region_text,
            "region": region,
            "n_units_metadata": safe_int(nwb.get("n_units")),
            "n_electrodes": safe_int(nwb.get("n_electrodes")),
        })

    return pd.DataFrame(rows)


def read_nwb_spike_times(path):
    try:
        from pynwb import NWBHDF5IO
    except Exception as e:
        raise RuntimeError("pynwb is required. Install with: python -m pip install pynwb") from e

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        if "spike_times" not in units_df.columns:
            return []
        spikes = []
        for st in units_df["spike_times"].values:
            arr = np.asarray(st, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            arr = arr[arr >= 0]
            if arr.size > 2:
                spikes.append(arr)
        return spikes


# =============================================================================
# Drift metric
# =============================================================================

def bin_spikes(spike_times, bin_size_s, fixed_units, random_seed, duration_window_s=None):
    """
    Returns bins x units count matrix.
    """
    cleaned = []
    for st in spike_times:
        arr = np.asarray(st, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        arr = arr[arr >= 0]
        if arr.size > 2:
            cleaned.append(arr)

    if len(cleaned) < fixed_units:
        return None, 0, 0

    rng = random.Random(random_seed)
    if len(cleaned) > fixed_units:
        cleaned = rng.sample(cleaned, fixed_units)

    t_min = min(float(np.min(x)) for x in cleaned)
    t_max = max(float(np.max(x)) for x in cleaned)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return None, len(cleaned), 0

    # Normalize start to zero
    cleaned = [x - t_min for x in cleaned]
    duration = t_max - t_min

    if duration_window_s is not None:
        duration = min(duration, duration_window_s)
        cleaned = [x[x <= duration] for x in cleaned]

    if duration < bin_size_s * 8:
        return None, len(cleaned), duration

    edges = np.arange(0, duration + bin_size_s, bin_size_s)
    if len(edges) < 8:
        return None, len(cleaned), duration

    matrix = np.zeros((len(edges) - 1, len(cleaned)), dtype=float)
    for j, st in enumerate(cleaned):
        matrix[:, j] = np.histogram(st, bins=edges)[0]

    return matrix, len(cleaned), duration


def cosine_distance_matrix(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 3:
        return None

    # Drop zero/constant units
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


def compute_toi(matrix, lag_unit, max_lag=20):
    dist = cosine_distance_matrix(matrix)
    if dist is None:
        return np.nan, pd.DataFrame(), None

    n = dist.shape[0]
    max_lag = min(max_lag, n - 1)
    rows = []
    for lag in range(1, max_lag + 1):
        vals = np.diag(dist, k=lag)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            rows.append({
                "lag_index": lag,
                "lag_value_s": lag * lag_unit,
                "mean_distance": float(np.mean(vals)),
                "median_distance": float(np.median(vals)),
                "n_pairs": int(len(vals)),
            })

    curve = pd.DataFrame(rows)
    if len(curve) < 3:
        return np.nan, curve, dist

    slope, intercept = np.polyfit(curve["lag_value_s"].values, curve["mean_distance"].values, 1)
    return float(slope), curve, dist


# =============================================================================
# Main analysis
# =============================================================================

def run_discovery(args):
    out = Path(args.output_dir)
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    meta = extract_nwb_metadata_table(args.nwb_json)

    # Strict focus on LEC and interpretable contexts
    keep_contexts = ["open_field", "object_context", "sequence_task", "sleep"]
    lec = meta[(meta["region"] == "LEC") & (meta["clean_context"].isin(keep_contexts))].copy()

    if args.max_files is not None:
        lec = lec.head(args.max_files)

    lec.to_csv(out / "lec_selected_sessions.csv", index=False)

    print("\nSelected LEC sessions by context:")
    print(lec["clean_context"].value_counts().to_string())

    session_results = []
    repeat_results = []
    all_curves = []
    examples_saved = 0

    for idx, row in lec.iterrows():
        path = Path(txt(row["absolute_path"]))
        sid = row["session_id"]
        ctx = row["clean_context"]

        print(f"\nSession: {sid} | {ctx}")

        if not path.exists():
            session_results.append({
                "session_id": sid,
                "subject_id": row["subject_id"],
                "context": ctx,
                "status": "missing_file",
            })
            continue

        try:
            spike_times = read_nwb_spike_times(path)
        except Exception as e:
            session_results.append({
                "session_id": sid,
                "subject_id": row["subject_id"],
                "context": ctx,
                "status": "read_error: " + repr(e),
            })
            continue

        if len(spike_times) < args.fixed_units:
            session_results.append({
                "session_id": sid,
                "subject_id": row["subject_id"],
                "context": ctx,
                "status": "insufficient_units",
                "n_units_available": len(spike_times),
            })
            continue

        # Repeated unit-count matched runs
        tois = []
        durations = []
        n_bins_list = []

        for rep in range(args.n_repeats):
            matrix, n_units_used, duration = bin_spikes(
                spike_times,
                bin_size_s=args.bin_size_s,
                fixed_units=args.fixed_units,
                random_seed=args.random_seed + rep,
                duration_window_s=args.duration_window_s,
            )
            if matrix is None:
                continue

            toi, curve, dist = compute_toi(matrix, lag_unit=args.bin_size_s, max_lag=args.max_lag)
            if np.isfinite(toi):
                tois.append(toi)
                durations.append(duration)
                n_bins_list.append(matrix.shape[0])

                repeat_results.append({
                    "session_id": sid,
                    "subject_id": row["subject_id"],
                    "context": ctx,
                    "repeat": rep,
                    "toi": toi,
                    "n_units_used": n_units_used,
                    "n_bins": matrix.shape[0],
                    "duration_s": duration,
                    "bin_size_s": args.bin_size_s,
                    "fixed_units": args.fixed_units,
                })

                if not curve.empty and rep == 0:
                    curve["session_id"] = sid
                    curve["subject_id"] = row["subject_id"]
                    curve["context"] = ctx
                    all_curves.append(curve)

                if dist is not None and examples_saved < args.n_examples and rep == 0:
                    plt.figure(figsize=(6, 5))
                    plt.imshow(dist, aspect="auto", origin="lower")
                    plt.colorbar(label="Cosine distance")
                    plt.xlabel("Time bin")
                    plt.ylabel("Time bin")
                    plt.title(f"LEC {ctx}")
                    plt.tight_layout()
                    plt.savefig(fig / f"example_distance_matrix_{examples_saved+1}_{ctx}.png", dpi=250)
                    plt.close()
                    examples_saved += 1

        if len(tois) == 0:
            session_results.append({
                "session_id": sid,
                "subject_id": row["subject_id"],
                "context": ctx,
                "status": "toi_failed",
                "n_units_available": len(spike_times),
            })
        else:
            session_results.append({
                "session_id": sid,
                "subject_id": row["subject_id"],
                "context": ctx,
                "status": "ok",
                "n_units_available": len(spike_times),
                "fixed_units": args.fixed_units,
                "n_repeats_ok": len(tois),
                "toi_median": float(np.median(tois)),
                "toi_mean": float(np.mean(tois)),
                "toi_std": float(np.std(tois)),
                "duration_s_median": float(np.median(durations)),
                "n_bins_median": float(np.median(n_bins_list)),
                "description": row["description"],
            })

    session_df = pd.DataFrame(session_results)
    repeat_df = pd.DataFrame(repeat_results)
    curves_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()

    session_df.to_csv(out / "lec_context_drift_session_summary.csv", index=False)
    repeat_df.to_csv(out / "lec_context_drift_repeated_subsampling.csv", index=False)
    curves_df.to_csv(out / "lec_context_distance_lag_curves.csv", index=False)

    stats_df, pairwise_df, controls_df = make_stats(session_df, repeat_df)
    stats_df.to_csv(out / "lec_context_drift_global_stats.csv", index=False)
    pairwise_df.to_csv(out / "lec_context_drift_pairwise_stats.csv", index=False)
    controls_df.to_csv(out / "lec_context_drift_controls.csv", index=False)

    make_figures(session_df, repeat_df, curves_df, pairwise_df, fig)
    write_report(out / "lec_context_drift_discovery_report.md", session_df, stats_df, pairwise_df, controls_df, args)

    print("\nDone.")
    print("Output:", out)
    print("\nUsable sessions:")
    ok = session_df[session_df["status"].eq("ok")]
    print(ok["context"].value_counts().to_string())


def make_stats(session_df, repeat_df):
    ok = session_df[session_df["status"].eq("ok")].copy()
    ok["toi_median"] = pd.to_numeric(ok["toi_median"], errors="coerce")

    groups = {ctx: sub["toi_median"].dropna().values for ctx, sub in ok.groupby("context")}
    h, p, names = kruskal_test(groups)
    stats_df = pd.DataFrame([{
        "test": "Kruskal-Wallis",
        "value": "toi_median",
        "groups": "; ".join(names),
        "H": h,
        "p_value": p,
        "n_sessions": int(ok["session_id"].nunique()),
    }])

    pairwise = []
    contexts = sorted(ok["context"].dropna().unique())
    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            c1 = contexts[i]
            c2 = contexts[j]
            a = ok.loc[ok["context"].eq(c1), "toi_median"]
            b = ok.loc[ok["context"].eq(c2), "toi_median"]
            pairwise.append({
                "contrast": f"{c1} vs {c2}",
                "group_a": c1,
                "group_b": c2,
                "median_a": float(pd.Series(a).median()) if len(a.dropna()) else np.nan,
                "median_b": float(pd.Series(b).median()) if len(b.dropna()) else np.nan,
                "cliffs_delta_a_minus_b": cliffs_delta(a, b),
                "p_value": mannwhitney_p(a, b),
                "n_a": int(len(a.dropna())),
                "n_b": int(len(b.dropna())),
            })

    pairwise_df = pd.DataFrame(pairwise)
    if not pairwise_df.empty:
        m = pairwise_df["p_value"].notna()
        pairwise_df.loc[m, "p_value_bh"] = bh_adjust(pairwise_df.loc[m, "p_value"].values)

    controls = []
    r, p_units = spearman_test(ok["n_units_available"], ok["toi_median"])
    controls.append({
        "control": "unit_count_available_vs_TOI",
        "spearman_r": r,
        "p_value": p_units,
        "interpretation": "Checks whether context effect may be confounded by available unit count."
    })
    r, p_dur = spearman_test(ok["duration_s_median"], ok["toi_median"])
    controls.append({
        "control": "duration_vs_TOI",
        "spearman_r": r,
        "p_value": p_dur,
        "interpretation": "Checks whether context effect may be confounded by recording duration."
    })
    r, p_bins = spearman_test(ok["n_bins_median"], ok["toi_median"])
    controls.append({
        "control": "n_bins_vs_TOI",
        "spearman_r": r,
        "p_value": p_bins,
        "interpretation": "Checks whether context effect may be confounded by number of time bins."
    })

    return stats_df, pairwise_df, pd.DataFrame(controls)


def make_figures(session_df, repeat_df, curves_df, pairwise_df, fig):
    ok = session_df[session_df["status"].eq("ok")].copy()
    ok["toi_median"] = pd.to_numeric(ok["toi_median"], errors="coerce")
    ok = ok.dropna(subset=["toi_median"])

    order = ["open_field", "object_context", "sequence_task", "sleep"]
    groups = []
    labels = []
    for ctx in order:
        vals = ok.loc[ok["context"].eq(ctx), "toi_median"].dropna().values
        if len(vals) >= 1:
            groups.append(vals)
            labels.append(ctx)

    if groups:
        plt.figure(figsize=(8.5, 5))
        try:
            plt.boxplot(groups, tick_labels=labels, showfliers=False)
        except TypeError:
            plt.boxplot(groups, labels=labels, showfliers=False)
        # overlay points
        for i, vals in enumerate(groups, start=1):
            jitter = np.random.default_rng(42).normal(i, 0.035, size=len(vals))
            plt.scatter(jitter, vals, alpha=0.75)
        plt.ylabel("TOI: distance-lag slope")
        plt.title("LEC temporal drift by behavioral context\n(unit-count matched)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(fig / "01_lec_toi_by_context_unit_matched.png", dpi=300)
        plt.close()

    if not curves_df.empty:
        plt.figure(figsize=(8.5, 5))
        for ctx in order:
            sub = curves_df[curves_df["context"].eq(ctx)]
            if sub.empty:
                continue
            mean_curve = sub.groupby("lag_value_s")["mean_distance"].mean().reset_index()
            plt.plot(mean_curve["lag_value_s"], mean_curve["mean_distance"], marker="o", label=ctx)
        plt.xlabel("Time lag (s)")
        plt.ylabel("Mean population distance")
        plt.title("LEC population distance increases with time lag")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig / "02_lec_distance_lag_curves_by_context.png", dpi=300)
        plt.close()

    # Pairwise effect size plot relative to open_field
    if not pairwise_df.empty:
        rel = pairwise_df[pairwise_df["contrast"].str.contains("open_field", na=False)].copy()
        if not rel.empty:
            labels = rel["contrast"].values
            vals = pd.to_numeric(rel["cliffs_delta_a_minus_b"], errors="coerce").values
            plt.figure(figsize=(8, 4))
            plt.barh(labels, vals)
            plt.axvline(0, linestyle="--")
            plt.xlabel("Cliff's delta")
            plt.title("Pairwise effect sizes involving open-field context")
            plt.tight_layout()
            plt.savefig(fig / "03_pairwise_effect_sizes_vs_open_field.png", dpi=300)
            plt.close()

    # Unit count/duration control scatter
    if not ok.empty:
        plt.figure(figsize=(6, 4))
        plt.scatter(ok["n_units_available"], ok["toi_median"], alpha=0.8)
        plt.xlabel("Available units")
        plt.ylabel("TOI median")
        plt.title("Control: unit count vs TOI")
        plt.tight_layout()
        plt.savefig(fig / "04_control_unit_count_vs_toi.png", dpi=300)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.scatter(ok["duration_s_median"] / 60, ok["toi_median"], alpha=0.8)
        plt.xlabel("Analyzed duration (min)")
        plt.ylabel("TOI median")
        plt.title("Control: duration vs TOI")
        plt.tight_layout()
        plt.savefig(fig / "05_control_duration_vs_toi.png", dpi=300)
        plt.close()

    # Context session counts
    if not ok.empty:
        counts = ok["context"].value_counts().reindex(order).dropna()
        plt.figure(figsize=(7, 4))
        plt.bar(counts.index, counts.values)
        plt.ylabel("Usable LEC sessions")
        plt.title("LEC sessions included by context")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(fig / "00_session_counts_by_context.png", dpi=300)
        plt.close()


def write_report(path, session_df, stats_df, pairwise_df, controls_df, args):
    ok = session_df[session_df["status"].eq("ok")].copy()
    ok["toi_median"] = pd.to_numeric(ok["toi_median"], errors="coerce")
    medians = ok.groupby("context")["toi_median"].median().sort_values(ascending=False)

    lines = [
        "# Discovery layer: context-dependent temporal drift in LEC",
        "",
        "## Biological question",
        "",
        "Does behavioral/event context modulate temporal drift in lateral entorhinal cortex population activity?",
        "",
        "## Method summary",
        "",
        f"- Region: LEC only.",
        f"- Spike-time bin size: {args.bin_size_s} s.",
        f"- Unit-count matched subsampling: {args.fixed_units} units per session.",
        f"- Repeats per session: {args.n_repeats}.",
        "- Metric: Temporal Organization Index (TOI), defined as the slope of mean population distance as a function of time lag.",
        "",
        "## Included sessions",
        "",
    ]

    for ctx, n in ok["context"].value_counts().items():
        lines.append(f"- {ctx}: {n} sessions")

    lines += [
        "",
        "## Median TOI by context",
        "",
    ]

    for ctx, val in medians.items():
        lines.append(f"- {ctx}: {val:.6g}")

    lines += [
        "",
        "## Global test",
        "",
    ]

    if not stats_df.empty:
        row = stats_df.iloc[0]
        lines.append(f"Kruskal-Wallis across contexts: H={row['H']:.4g}, p={row['p_value']:.4g}.")

    lines += [
        "",
        "## Controls",
        "",
    ]

    for _, row in controls_df.iterrows():
        lines.append(f"- {row['control']}: Spearman r={row['spearman_r']:.4g}, p={row['p_value']:.4g}")

    lines += [
        "",
        "## Suggested interpretation",
        "",
        "This analysis provides the discovery-style layer of the project. It tests a concrete neuroscience hypothesis within the strongest raw dataset identified by the cross-dataset reuse pipeline: whether temporal organization of LEC population activity depends on behavioral context. If sequence/object contexts show higher TOI than open-field sessions after unit-count matching, the result supports the idea that event-structured experience is associated with stronger temporal-context drift in LEC.",
        "",
        "## Caution",
        "",
        "This does not prove causality and does not replace the original full analysis pipeline of the source paper. It is a simplified, reusable spike-time analysis designed to demonstrate that metadata-guided dataset selection can lead to a biologically meaningful secondary analysis.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run NWB LEC context drift discovery layer.")
    parser.add_argument("--nwb_json", required=True)
    parser.add_argument("--output_dir", default="outputs/nwb_lec_context_drift_discovery")
    parser.add_argument("--bin_size_s", type=float, default=10.0)
    parser.add_argument("--fixed_units", type=int, default=50)
    parser.add_argument("--n_repeats", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--max_lag", type=int, default=20)
    parser.add_argument("--duration_window_s", type=float, default=None,
                        help="Optional fixed analysis duration in seconds. Example: 1200 for 20 min.")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_examples", type=int, default=4)
    args = parser.parse_args()

    run_discovery(args)


if __name__ == "__main__":
    main()
