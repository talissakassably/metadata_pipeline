# -*- coding: utf-8 -*-
"""
Actual cross-dataset CA1 unit-activity case study.

This is different from the metadata-only case study:
it tries to load the ACTUAL data files and extract unit/spike-train summaries.

Case study question
-------------------
Can reusable CA1-related datasets be compared through actual unit-activity
summary features, such as spike counts, firing rates, unit yield and recording
duration?

Supported data sources
----------------------
1. NWB files:
   - Uses pynwb if installed.
   - Extracts units spike_times.

2. Neo / pickle files:
   - Tries pickle loading.
   - Extracts Neo Block/Segment spiketrains if possible.
   - Also handles simple dictionaries/lists recursively when possible.

3. MClust .t files:
   - Parses MClust timestamp files from openfield-like datasets when present.
   - Each .t file is treated as one unit.

Outputs
-------
outputs/cross_dataset_unit_case_study/
    ca1_unit_activity_table.csv
    ca1_session_activity_summary.csv
    ca1_unit_activity_extraction_log.csv
    ca1_unit_activity_dataset_summary.csv
    ca1_unit_activity_kruskal_tests.csv
    ca1_unit_activity_pca_coordinates.csv
    ca1_unit_activity_pca_loadings.csv
    ca1_unit_activity_case_study_summary.md

outputs/figures/
    12_unit_firing_rate_by_dataset.png
    13_unit_yield_by_dataset.png
    14_session_median_firing_rate_by_dataset.png
    15_unit_activity_pca.png
    16_unit_activity_pca_top_drivers.png

Run from repository root
------------------------
py meta_analysis\\run_cross_dataset_unit_activity_case_study.py `
  --openfield_root "C:\\Users\\tkassably\\Downloads\\d-40faae41-7e72-4c3c-9abf-91dea149158d" `
  --nwb_root "C:\\Users\\tkassably\\Downloads\\d-885b4936-9345-43bd-880e-eebc19898ded" `
  --legacy_root "C:\\Users\\tkassably\\Downloads\\d-d406a98c-ae5c-4fb3-9f0c-4cf4de9b1094-hbp-data-002061-code" `
  --touchandsee_root "C:\\Users\\tkassably\\Downloads\\p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681" `
  --output_dir meta_analysis\\outputs
"""

import argparse
import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        if hasattr(value, "rescale"):
            return float(value.rescale("s").magnitude)
        if hasattr(value, "magnitude"):
            arr = np.asarray(value.magnitude)
            if arr.size == 1:
                return float(arr)
            return default
        return float(value)
    except Exception:
        return default


def safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0


def infer_session_id_from_path(path):
    path = Path(path)
    return path.stem


def infer_subject_from_path(path):
    """
    Weak heuristic: tries to infer animal/subject from filename or parent folders.
    """
    path = Path(path)
    text = " ".join([p.name for p in path.parents[:4]] + [path.stem])
    for token in text.replace("-", "_").replace(".", "_").split("_"):
        lower = token.lower()
        if lower.startswith("rat") or lower.startswith("mouse") or lower.startswith("animal"):
            return token
        if lower in ["ramachandran", "taskmaster", "turing"]:
            return token
    return ""


def compute_duration_from_spike_times(times):
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]

    if len(times) == 0:
        return np.nan

    duration = float(times.max() - times.min())

    if duration <= 0:
        return np.nan

    return duration


def make_unit_row(dataset, session_id, unit_id, n_spikes, duration_s, source_file, extraction_method, subject_id="", brain_region="", behavioral_context=""):
    n_spikes = int(n_spikes) if not pd.isna(n_spikes) else 0
    duration_s = safe_float(duration_s, default=np.nan)

    if pd.isna(duration_s) or duration_s <= 0:
        firing_rate_hz = np.nan
    else:
        firing_rate_hz = n_spikes / duration_s

    return {
        "dataset_short_name": dataset,
        "session_id": str(session_id),
        "subject_id": str(subject_id) if subject_id is not None else "",
        "unit_id": str(unit_id),
        "n_spikes": n_spikes,
        "duration_s": duration_s,
        "mean_firing_rate_hz": firing_rate_hz,
        "brain_region": brain_region or "",
        "behavioral_context": behavioral_context or "",
        "source_file": str(source_file),
        "extraction_method": extraction_method,
    }


# ---------------------------------------------------------------------
# MClust .t file parser
# ---------------------------------------------------------------------

def read_mclust_t_file(path):
    """
    Read MClust .t timestamp files.

    MClust .t files usually contain an ASCII header ending with %%ENDHEADER,
    followed by big-endian uint32 timestamps. Timestamp unit is commonly 0.1 ms.
    We convert timestamps to seconds with ts * 1e-4.

    If parsing fails, returns None.
    """
    path = Path(path)

    try:
        data = path.read_bytes()
    except Exception:
        return None

    marker = b"%%ENDHEADER"
    idx = data.find(marker)

    if idx == -1:
        # Some files may have no header; try pure binary.
        binary = data
    else:
        after = idx + len(marker)
        # Skip newline characters after header.
        while after < len(data) and data[after] in [10, 13]:
            after += 1
        binary = data[after:]

    if len(binary) < 4:
        return None

    # Read as big-endian uint32.
    n = len(binary) // 4
    binary = binary[: n * 4]

    try:
        timestamps = np.frombuffer(binary, dtype=">u4").astype(float)
    except Exception:
        return None

    # Convert 0.1 ms timestamps to seconds.
    times_s = timestamps * 1e-4

    # Remove pathological zeros if all zero.
    times_s = times_s[np.isfinite(times_s)]

    if len(times_s) == 0:
        return None

    return times_s


def extract_openfield_mclust_units(root):
    root = Path(root)
    rows = []
    logs = []

    if not root.exists():
        return rows, [{"dataset_short_name": "openfield_ca1", "file": str(root), "status": "missing_root", "message": "Root folder not found"}]

    # MClust .t files can be many; exclude text/code files and hidden folders.
    t_files = sorted(root.rglob("*.t"))

    if len(t_files) == 0:
        logs.append({
            "dataset_short_name": "openfield_ca1",
            "file": str(root),
            "status": "no_t_files",
            "message": "No MClust .t files found",
        })
        return rows, logs

    for path in t_files:
        # Avoid possible non-unit files with tiny names? Keep all .t files initially.
        try:
            times_s = read_mclust_t_file(path)

            if times_s is None or len(times_s) == 0:
                logs.append({
                    "dataset_short_name": "openfield_ca1",
                    "file": str(path),
                    "status": "failed",
                    "message": "Could not parse .t file",
                })
                continue

            session_id = path.parent.name
            subject_id = infer_subject_from_path(path)
            duration_s = compute_duration_from_spike_times(times_s)

            rows.append(make_unit_row(
                dataset="openfield_ca1",
                session_id=session_id,
                subject_id=subject_id,
                unit_id=path.stem,
                n_spikes=len(times_s),
                duration_s=duration_s,
                source_file=path,
                extraction_method="mclust_t_file",
                behavioral_context="open field / navigation",
            ))

            logs.append({
                "dataset_short_name": "openfield_ca1",
                "file": str(path),
                "status": "success",
                "message": f"Parsed {len(times_s)} timestamps",
            })

        except Exception as error:
            logs.append({
                "dataset_short_name": "openfield_ca1",
                "file": str(path),
                "status": "failed",
                "message": repr(error),
            })

    return rows, logs


# ---------------------------------------------------------------------
# NWB extraction
# ---------------------------------------------------------------------

def extract_nwb_units(root):
    root = Path(root)
    rows = []
    logs = []

    if not root.exists():
        return rows, [{"dataset_short_name": "nwb_ca1", "file": str(root), "status": "missing_root", "message": "Root folder not found"}]

    nwb_files = sorted(root.rglob("*.nwb"))

    if len(nwb_files) == 0:
        logs.append({
            "dataset_short_name": "nwb_ca1",
            "file": str(root),
            "status": "no_nwb_files",
            "message": "No NWB files found",
        })
        return rows, logs

    try:
        from pynwb import NWBHDF5IO
    except Exception as error:
        logs.append({
            "dataset_short_name": "nwb_ca1",
            "file": str(root),
            "status": "missing_dependency",
            "message": f"pynwb not available: {repr(error)}. Install with: py -m pip install pynwb",
        })
        return rows, logs

    for path in nwb_files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io = NWBHDF5IO(str(path), "r", load_namespaces=True)
                nwbfile = io.read()

            session_id = getattr(nwbfile, "identifier", None) or path.stem
            subject_id = ""
            if getattr(nwbfile, "subject", None) is not None:
                subject_id = getattr(nwbfile.subject, "subject_id", "") or ""

            brain_regions = []
            try:
                electrodes = nwbfile.electrodes.to_dataframe()
                if "location" in electrodes.columns:
                    brain_regions = sorted(set(str(x) for x in electrodes["location"].dropna().unique()))
            except Exception:
                pass
            brain_region = "; ".join(brain_regions)

            behavioral_context = getattr(nwbfile, "session_description", "") or "NWB CA1 electrophysiology"

            if nwbfile.units is None:
                logs.append({
                    "dataset_short_name": "nwb_ca1",
                    "file": str(path),
                    "status": "no_units",
                    "message": "NWB file has no units table",
                })
                try:
                    io.close()
                except Exception:
                    pass
                continue

            units_df = nwbfile.units.to_dataframe()

            if "spike_times" not in units_df.columns:
                logs.append({
                    "dataset_short_name": "nwb_ca1",
                    "file": str(path),
                    "status": "no_spike_times",
                    "message": "NWB units table has no spike_times column",
                })
                try:
                    io.close()
                except Exception:
                    pass
                continue

            for unit_index, unit_row in units_df.iterrows():
                spike_times = unit_row["spike_times"]

                if spike_times is None:
                    continue

                spike_times = np.asarray(spike_times, dtype=float)
                n_spikes = len(spike_times)
                duration_s = compute_duration_from_spike_times(spike_times)

                rows.append(make_unit_row(
                    dataset="nwb_ca1",
                    session_id=session_id,
                    subject_id=subject_id,
                    unit_id=unit_index,
                    n_spikes=n_spikes,
                    duration_s=duration_s,
                    source_file=path,
                    extraction_method="nwb_units_spike_times",
                    brain_region=brain_region,
                    behavioral_context=behavioral_context,
                ))

            logs.append({
                "dataset_short_name": "nwb_ca1",
                "file": str(path),
                "status": "success",
                "message": f"Extracted {len(units_df)} units",
            })

            try:
                io.close()
            except Exception:
                pass

        except Exception as error:
            logs.append({
                "dataset_short_name": "nwb_ca1",
                "file": str(path),
                "status": "failed",
                "message": repr(error),
            })

    return rows, logs


# ---------------------------------------------------------------------
# Pickle / Neo extraction
# ---------------------------------------------------------------------

def extract_spiketrains_from_neo_block(obj, source_file, dataset, session_id, subject_id="", behavioral_context=""):
    rows = []

    # Neo Block has .segments
    segments = getattr(obj, "segments", None)

    if segments is None:
        return rows

    unit_counter = 0

    for seg_idx, segment in enumerate(segments):
        spiketrains = getattr(segment, "spiketrains", []) or []

        for st_idx, st in enumerate(spiketrains):
            try:
                n_spikes = len(st)

                # Duration: prefer t_start/t_stop, otherwise use spike time range.
                t_start = safe_float(getattr(st, "t_start", None), default=np.nan)
                t_stop = safe_float(getattr(st, "t_stop", None), default=np.nan)

                if not pd.isna(t_start) and not pd.isna(t_stop) and t_stop > t_start:
                    duration_s = t_stop - t_start
                else:
                    try:
                        times = np.asarray(st.rescale("s").magnitude, dtype=float)
                    except Exception:
                        try:
                            times = np.asarray(st, dtype=float)
                        except Exception:
                            times = np.array([])
                    duration_s = compute_duration_from_spike_times(times)

                annotations = getattr(st, "annotations", {}) or {}
                brain_region = (
                    annotations.get("brain_region")
                    or annotations.get("region")
                    or annotations.get("area")
                    or annotations.get("recording_group")
                    or ""
                )

                unit_name = getattr(st, "name", None) or f"seg{seg_idx}_unit{st_idx}"

                rows.append(make_unit_row(
                    dataset=dataset,
                    session_id=session_id,
                    subject_id=subject_id,
                    unit_id=unit_name or unit_counter,
                    n_spikes=n_spikes,
                    duration_s=duration_s,
                    source_file=source_file,
                    extraction_method="neo_block_spiketrain",
                    brain_region=str(brain_region),
                    behavioral_context=behavioral_context,
                ))

                unit_counter += 1

            except Exception:
                continue

    return rows


def recursively_find_spiketrains(obj, max_depth=5):
    """
    Best-effort recursive extraction for dict/list objects that contain arrays
    of spike times. This handles simple custom pickle structures.
    """
    found = []

    def visit(x, path, depth):
        if depth > max_depth:
            return

        # Neo-like spiketrain object
        if hasattr(x, "t_start") and hasattr(x, "t_stop") and hasattr(x, "__len__"):
            found.append((path, x))
            return

        # Dict
        if isinstance(x, dict):
            for k, v in x.items():
                key = str(k).lower()
                if "spike" in key and isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.asarray(v, dtype=object)
                    if arr.ndim == 1 and len(arr) > 0:
                        # If it looks numeric, treat as one spiketrain.
                        try:
                            numeric = np.asarray(v, dtype=float)
                            if numeric.size > 0:
                                found.append((path + "/" + str(k), numeric))
                        except Exception:
                            pass
                visit(v, path + "/" + str(k), depth + 1)
            return

        # List/tuple
        if isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                visit(v, path + f"[{i}]", depth + 1)
            return

    visit(obj, "root", 0)
    return found


def extract_pickle_units(root, dataset, behavioral_context):
    root = Path(root)
    rows = []
    logs = []

    if not root.exists():
        return rows, [{"dataset_short_name": dataset, "file": str(root), "status": "missing_root", "message": "Root folder not found"}]

    pickle_files = sorted(list(root.rglob("*.pkl")) + list(root.rglob("*.pickle")))

    if len(pickle_files) == 0:
        logs.append({
            "dataset_short_name": dataset,
            "file": str(root),
            "status": "no_pickle_files",
            "message": "No .pkl/.pickle files found",
        })
        return rows, logs

    for path in pickle_files:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)

            session_id = path.stem
            subject_id = infer_subject_from_path(path)

            # First try Neo Block extraction
            neo_rows = extract_spiketrains_from_neo_block(
                obj,
                source_file=path,
                dataset=dataset,
                session_id=session_id,
                subject_id=subject_id,
                behavioral_context=behavioral_context,
            )

            if len(neo_rows) > 0:
                rows.extend(neo_rows)
                logs.append({
                    "dataset_short_name": dataset,
                    "file": str(path),
                    "status": "success",
                    "message": f"Extracted {len(neo_rows)} Neo spiketrains",
                })
                continue

            # Try recursive custom structure extraction
            found = recursively_find_spiketrains(obj)

            custom_count = 0
            for unit_idx, (unit_path, st) in enumerate(found):
                try:
                    if hasattr(st, "rescale"):
                        times = np.asarray(st.rescale("s").magnitude, dtype=float)
                    else:
                        times = np.asarray(st, dtype=float)

                    if len(times) == 0:
                        continue

                    duration_s = compute_duration_from_spike_times(times)

                    rows.append(make_unit_row(
                        dataset=dataset,
                        session_id=session_id,
                        subject_id=subject_id,
                        unit_id=f"{unit_idx}_{unit_path}",
                        n_spikes=len(times),
                        duration_s=duration_s,
                        source_file=path,
                        extraction_method="recursive_pickle_spike_times",
                        behavioral_context=behavioral_context,
                    ))
                    custom_count += 1
                except Exception:
                    continue

            if custom_count > 0:
                logs.append({
                    "dataset_short_name": dataset,
                    "file": str(path),
                    "status": "success",
                    "message": f"Extracted {custom_count} recursive spike-time arrays",
                })
            else:
                logs.append({
                    "dataset_short_name": dataset,
                    "file": str(path),
                    "status": "no_spiketrains_found",
                    "message": "Pickle loaded but no spiketrains found",
                })

        except Exception as error:
            logs.append({
                "dataset_short_name": dataset,
                "file": str(path),
                "status": "failed",
                "message": repr(error),
            })

    return rows, logs


# ---------------------------------------------------------------------
# Summaries and analyses
# ---------------------------------------------------------------------

def make_session_summary(unit_df):
    if unit_df.empty:
        return pd.DataFrame()

    grouped = unit_df.groupby(["dataset_short_name", "session_id"], dropna=False)

    rows = []

    for (dataset, session_id), group in grouped:
        firing = pd.to_numeric(group["mean_firing_rate_hz"], errors="coerce")
        spikes = pd.to_numeric(group["n_spikes"], errors="coerce").fillna(0)
        duration = pd.to_numeric(group["duration_s"], errors="coerce")

        rows.append({
            "dataset_short_name": dataset,
            "session_id": session_id,
            "subject_id": "; ".join(sorted(set(str(x) for x in group["subject_id"] if str(x).strip()))),
            "behavioral_context": "; ".join(sorted(set(str(x) for x in group["behavioral_context"] if str(x).strip()))),
            "brain_region": "; ".join(sorted(set(str(x) for x in group["brain_region"] if str(x).strip()))),
            "n_units": len(group),
            "total_spikes": int(spikes.sum()),
            "median_firing_rate_hz": float(firing.median(skipna=True)),
            "mean_firing_rate_hz": float(firing.mean(skipna=True)),
            "std_firing_rate_hz": float(firing.std(skipna=True)),
            "active_unit_fraction_0_1hz": float((firing > 0.1).mean()),
            "active_unit_fraction_1hz": float((firing > 1.0).mean()),
            "median_duration_s": float(duration.median(skipna=True)),
            "min_duration_s": float(duration.min(skipna=True)),
            "max_duration_s": float(duration.max(skipna=True)),
            "extraction_methods": "; ".join(sorted(set(str(x) for x in group["extraction_method"]))),
        })

    return pd.DataFrame(rows)


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)

    if n == 0:
        return np.array([])

    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    cumulative = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        cumulative = min(cumulative, value)
        adjusted[order[i]] = cumulative

    return np.minimum(adjusted, 1.0)


def kruskal_by_dataset(df, features, level_name):
    try:
        from scipy.stats import kruskal
    except Exception:
        return pd.DataFrame()

    rows = []

    for feature in features:
        if feature not in df.columns:
            continue

        groups = []
        names = []

        for dataset, sub in df.groupby("dataset_short_name"):
            values = pd.to_numeric(sub[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values

            if len(values) >= 2:
                groups.append(values)
                names.append(dataset)

        if len(groups) < 2:
            continue

        try:
            stat, p_value = kruskal(*groups)
        except Exception:
            continue

        n = sum(len(g) for g in groups)
        k = len(groups)
        eta = (stat - k + 1) / (n - k) if n > k else np.nan
        eta = max(0, eta) if not pd.isna(eta) else np.nan

        rows.append({
            "level": level_name,
            "feature": feature,
            "groups": "; ".join(names),
            "n_observations": n,
            "kruskal_H": stat,
            "p_value": p_value,
            "eta_squared_approx": eta,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["p_value_bh"] = benjamini_hochberg(out["p_value"].values)
        out = out.sort_values(["p_value_bh", "eta_squared_approx"], ascending=[True, False])

    return out


def run_session_pca(session_df):
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    features = [
        "n_units",
        "total_spikes",
        "median_firing_rate_hz",
        "mean_firing_rate_hz",
        "std_firing_rate_hz",
        "active_unit_fraction_0_1hz",
        "active_unit_fraction_1hz",
        "median_duration_s",
    ]

    X = pd.DataFrame(index=session_df.index)

    for f in features:
        if f in session_df.columns:
            values = pd.to_numeric(session_df[f], errors="coerce").fillna(0)
            if f in ["n_units", "total_spikes", "median_firing_rate_hz", "mean_firing_rate_hz", "std_firing_rate_hz", "median_duration_s"]:
                values = np.log1p(values.clip(lower=0))
            X[f] = values

    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.shape[1] < 2 or len(X) < 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    values = X.values.astype(float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1
    X_scaled = (values - mean) / std

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    coord_df = pd.DataFrame({
        "pca_1": coords[:, 0],
        "pca_2": coords[:, 1],
    })

    for col in ["dataset_short_name", "session_id", "behavioral_context", "n_units", "median_firing_rate_hz", "total_spikes"]:
        if col in session_df.columns:
            coord_df[col] = session_df[col].values

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_loading", "PC2_loading"],
        index=X.columns,
    ).reset_index(names="feature")

    loadings["abs_PC1_loading"] = loadings["PC1_loading"].abs()
    loadings["abs_PC2_loading"] = loadings["PC2_loading"].abs()
    loadings["max_abs_loading"] = loadings[["abs_PC1_loading", "abs_PC2_loading"]].max(axis=1)

    explained = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })

    return coord_df, loadings, explained


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def plot_unit_firing_rates(unit_df, figures_dir):
    if unit_df.empty:
        return

    datasets = sorted(unit_df["dataset_short_name"].unique())
    data = []

    for dataset in datasets:
        values = pd.to_numeric(
            unit_df.loc[unit_df["dataset_short_name"] == dataset, "mean_firing_rate_hz"],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        values = values[values >= 0]
        # log10(Hz + small value) for readability.
        data.append(np.log10(values + 1e-3))

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=datasets, showfliers=False)
    plt.ylabel("log10(mean firing rate Hz + 0.001)")
    plt.title("Actual data case study: unit firing-rate distributions")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "12_unit_firing_rate_by_dataset.png", dpi=250)
    plt.close()


def plot_unit_yield(session_df, figures_dir):
    if session_df.empty:
        return

    datasets = sorted(session_df["dataset_short_name"].unique())
    data = []

    for dataset in datasets:
        values = pd.to_numeric(
            session_df.loc[session_df["dataset_short_name"] == dataset, "n_units"],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        data.append(values)

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=datasets, showfliers=False)
    plt.ylabel("Number of extracted units per session")
    plt.title("Actual data case study: unit yield per session")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "13_unit_yield_by_dataset.png", dpi=250)
    plt.close()


def plot_session_median_firing(session_df, figures_dir):
    if session_df.empty:
        return

    datasets = sorted(session_df["dataset_short_name"].unique())
    data = []

    for dataset in datasets:
        values = pd.to_numeric(
            session_df.loc[session_df["dataset_short_name"] == dataset, "median_firing_rate_hz"],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        data.append(np.log10(values + 1e-3))

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=datasets, showfliers=False)
    plt.ylabel("log10(session median firing rate Hz + 0.001)")
    plt.title("Actual data case study: session median firing rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "14_session_median_firing_rate_by_dataset.png", dpi=250)
    plt.close()


def plot_activity_pca(coord_df, figures_dir):
    if coord_df.empty:
        return

    plt.figure(figsize=(8, 6))

    for dataset in sorted(coord_df["dataset_short_name"].unique()):
        sub = coord_df[coord_df["dataset_short_name"] == dataset]
        plt.scatter(sub["pca_1"], sub["pca_2"], label=dataset, alpha=0.75)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Actual data case study: PCA of session unit-activity summaries")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "15_unit_activity_pca.png", dpi=250)
    plt.close()


def plot_pca_drivers(loadings, figures_dir):
    if loadings.empty:
        return

    top = loadings.sort_values("max_abs_loading", ascending=False).head(10).copy()
    top = top.sort_values("max_abs_loading", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(top["feature"], top["max_abs_loading"])
    plt.xlabel("Maximum absolute loading on PC1 or PC2")
    plt.title("Actual data case study: top unit-activity PCA drivers")
    plt.tight_layout()
    plt.savefig(figures_dir / "16_unit_activity_pca_top_drivers.png", dpi=250)
    plt.close()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def write_summary(out_path, unit_df, session_df, logs_df, unit_tests, session_tests, pca_explained):
    lines = []

    lines.append("# Actual cross-dataset CA1 unit-activity case study\n")
    lines.append("## Case study question\n")
    lines.append(
        "Can CA1-related datasets be compared through actual extracted unit/spike-train "
        "activity summaries such as spike counts, firing rates, unit yield and recording duration?\n"
    )

    lines.append("## Data extracted\n")
    lines.append(f"- Unit-level rows extracted: {len(unit_df)}")
    lines.append(f"- Session-level rows extracted: {len(session_df)}")

    if not unit_df.empty:
        for dataset, sub in unit_df.groupby("dataset_short_name"):
            lines.append(f"- {dataset}: {len(sub)} units from {sub['session_id'].nunique()} sessions/files")
    lines.append("")

    if not logs_df.empty:
        success = int((logs_df["status"] == "success").sum())
        failed = int((logs_df["status"] != "success").sum())
        lines.append("## Extraction log\n")
        lines.append(f"- Successful files: {success}")
        lines.append(f"- Non-success/failed files: {failed}")
        lines.append("")

    if not unit_tests.empty:
        lines.append("## Unit-level dataset differences\n")
        for _, row in unit_tests.head(6).iterrows():
            lines.append(
                f"- {row['feature']}: Kruskal-Wallis H={row['kruskal_H']:.2f}, "
                f"BH-adjusted p={row['p_value_bh']:.3g}, effect≈{row['eta_squared_approx']:.2f}"
            )
        lines.append("")

    if not session_tests.empty:
        lines.append("## Session-level dataset differences\n")
        for _, row in session_tests.head(6).iterrows():
            lines.append(
                f"- {row['feature']}: Kruskal-Wallis H={row['kruskal_H']:.2f}, "
                f"BH-adjusted p={row['p_value_bh']:.3g}, effect≈{row['eta_squared_approx']:.2f}"
            )
        lines.append("")

    if not pca_explained.empty:
        lines.append("## PCA of session activity summaries\n")
        for _, row in pca_explained.iterrows():
            lines.append(f"- {row['component']}: explains {row['explained_variance_ratio'] * 100:.1f}% of variance")
        lines.append("")

    lines.append("## Interpretation\n")
    lines.append(
        "This analysis is a real data-level case study: it extracts unit/spike-train "
        "summaries from the underlying electrophysiology files when readable, then "
        "compares firing-rate and unit-yield profiles across datasets. Metadata is used "
        "as context and for harmonization, but the central variables come from actual "
        "spike/unit data rather than metadata completeness alone."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_case_study(openfield_root, nwb_root, legacy_root, touchandsee_root, output_dir):
    output_dir = Path(output_dir)
    case_dir = output_dir / "actual_data_case_study"
    figures_dir = output_dir / "figures"

    case_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_logs = []

    # Openfield MClust .t files
    rows, logs = extract_openfield_mclust_units(openfield_root)
    all_rows.extend(rows)
    all_logs.extend(logs)

    # NWB units
    rows, logs = extract_nwb_units(nwb_root)
    all_rows.extend(rows)
    all_logs.extend(logs)

    # Legacy touchscreen pickles
    rows, logs = extract_pickle_units(
        legacy_root,
        dataset="legacy_touchscreen",
        behavioral_context="touchscreen / tactile-visual task",
    )
    all_rows.extend(rows)
    all_logs.extend(logs)

    # TouchAndSee pickles
    rows, logs = extract_pickle_units(
        touchandsee_root,
        dataset="touchandsee",
        behavioral_context="TouchAndSee",
    )
    all_rows.extend(rows)
    all_logs.extend(logs)

    unit_df = pd.DataFrame(all_rows)
    logs_df = pd.DataFrame(all_logs)

    if unit_df.empty:
        logs_df.to_csv(case_dir / "ca1_unit_activity_extraction_log.csv", index=False)
        raise RuntimeError(
            "No unit/spike-train data could be extracted. "
            "Check dependencies: pynwb for NWB, compatible Neo/pickle environment for .pkl, "
            "and MClust .t files for openfield."
        )

    # Clean unit table
    unit_df["n_spikes"] = pd.to_numeric(unit_df["n_spikes"], errors="coerce").fillna(0)
    unit_df["duration_s"] = pd.to_numeric(unit_df["duration_s"], errors="coerce")
    unit_df["mean_firing_rate_hz"] = pd.to_numeric(unit_df["mean_firing_rate_hz"], errors="coerce")

    unit_df.to_csv(case_dir / "ca1_unit_activity_table.csv", index=False)
    logs_df.to_csv(case_dir / "ca1_unit_activity_extraction_log.csv", index=False)

    session_df = make_session_summary(unit_df)
    session_df.to_csv(case_dir / "ca1_session_activity_summary.csv", index=False)

    dataset_summary = session_df.groupby("dataset_short_name").agg(
        n_sessions=("session_id", "nunique"),
        total_units=("n_units", "sum"),
        median_units_per_session=("n_units", "median"),
        median_session_firing_rate_hz=("median_firing_rate_hz", "median"),
        median_total_spikes=("total_spikes", "median"),
        median_duration_s=("median_duration_s", "median"),
    ).reset_index()
    dataset_summary.to_csv(case_dir / "ca1_unit_activity_dataset_summary.csv", index=False)

    # Statistical tests
    unit_tests = kruskal_by_dataset(
        unit_df,
        ["n_spikes", "duration_s", "mean_firing_rate_hz"],
        level_name="unit",
    )
    unit_tests.to_csv(case_dir / "ca1_unit_activity_unit_level_kruskal_tests.csv", index=False)

    session_tests = kruskal_by_dataset(
        session_df,
        [
            "n_units",
            "total_spikes",
            "median_firing_rate_hz",
            "mean_firing_rate_hz",
            "active_unit_fraction_0_1hz",
            "active_unit_fraction_1hz",
            "median_duration_s",
        ],
        level_name="session",
    )
    session_tests.to_csv(case_dir / "ca1_unit_activity_session_level_kruskal_tests.csv", index=False)

    # PCA
    pca_coords, pca_loadings, pca_explained = run_session_pca(session_df)
    pca_coords.to_csv(case_dir / "ca1_unit_activity_pca_coordinates.csv", index=False)
    pca_loadings.to_csv(case_dir / "ca1_unit_activity_pca_loadings.csv", index=False)
    pca_explained.to_csv(case_dir / "ca1_unit_activity_pca_explained_variance.csv", index=False)

    # Figures
    plot_unit_firing_rates(unit_df, figures_dir)
    plot_unit_yield(session_df, figures_dir)
    plot_session_median_firing(session_df, figures_dir)
    plot_activity_pca(pca_coords, figures_dir)
    plot_pca_drivers(pca_loadings, figures_dir)

    write_summary(
        case_dir / "ca1_actual_data_case_study_summary.md",
        unit_df,
        session_df,
        logs_df,
        unit_tests,
        session_tests,
        pca_explained,
    )

    print("Done.")
    print("Actual data case study outputs:", case_dir)
    print("Figures:", figures_dir)
    print("Unit rows:", len(unit_df))
    print("Session rows:", len(session_df))
    print("\nUnits by dataset:")
    print(unit_df.groupby("dataset_short_name").size().to_string())

    print("\nSessions by dataset:")
    print(session_df.groupby("dataset_short_name").size().to_string())

    if not unit_tests.empty:
        print("\nTop unit-level differences:")
        print(unit_tests.head(5).to_string(index=False))

    if not session_tests.empty:
        print("\nTop session-level differences:")
        print(session_tests.head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Run actual cross-dataset CA1 unit-activity case study."
    )

    parser.add_argument("--openfield_root", required=True)
    parser.add_argument("--nwb_root", required=True)
    parser.add_argument("--legacy_root", required=True)
    parser.add_argument("--touchandsee_root", required=True)
    parser.add_argument("--output_dir", default="meta_analysis/outputs")

    args = parser.parse_args()

    run_case_study(
        openfield_root=args.openfield_root,
        nwb_root=args.nwb_root,
        legacy_root=args.legacy_root,
        touchandsee_root=args.touchandsee_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
