# -*- coding: utf-8 -*-

"""
TouchAndSee internal metadata extractor.

Dataset:
    p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681

Purpose:
    Load old Neo pickle files and extract internal session-level metadata:
    - subject / session identity
    - Neo Block / Segment counts
    - spiketrains and total spikes
    - analog signals / LFP-like signals
    - events and epochs
    - annotations keys, but not raw arrays
    - compact quality / completeness flags

This script intentionally does NOT store:
    - raw signal values
    - spike times
    - waveform arrays
    - full annotations if they are large arrays/lists

Recommended environment:
    conda create -n touchandsee_old python=3.9 -y
    conda activate touchandsee_old
    python -m pip install "numpy==1.26.4" "neo==0.13.4" quantities pandas scipy
"""

import os
import re
import sys
import json
import pickle
import argparse
import traceback
import inspect
from pathlib import Path
from datetime import datetime


LARGE_FIELD_KEYWORDS = [
    "waveform",
    "waveforms",
    "signal",
    "signals",
    "spike_times",
    "times",
    "data",
    "array",
    "template",
    "templates",
]


def patch_old_neo_pickle_compatibility(verbose=True):
    """
    Compatibility patch for old Neo pickles.

    Old TouchAndSee pickles can contain:
      - annotations=None
      - array_annotations=None
      - copy argument from older Neo reconstruction

    Newer Neo versions can fail on these. This patch normalizes them before
    reconstruction.
    """

    applied = []

    try:
        import neo.core.analogsignal as analogsignal

        # Patch reconstruction functions used by pickle.
        for function_name in [
            "_new_AnalogSignalArray",
            "_new_AnalogSignal",
        ]:
            if not hasattr(analogsignal, function_name):
                continue

            original_function = getattr(analogsignal, function_name)
            signature = inspect.signature(original_function)

            def make_patched_function(original_function, signature):
                def patched_function(*args, **kwargs):
                    bound = signature.bind_partial(*args, **kwargs)

                    if "annotations" in signature.parameters:
                        if bound.arguments.get("annotations") is None:
                            bound.arguments["annotations"] = {}

                    if "array_annotations" in signature.parameters:
                        # Old Neo pickles can contain scalar / wrong-length array annotations.
                        # For metadata extraction we do not need these per-sample/channel annotations,
                        # so drop them to avoid: Incorrect length of array annotation.
                        bound.arguments["array_annotations"] = {}

                    if "copy" in signature.parameters:
                        # Prevent Neo/Numpy2 copy deprecation errors.
                        bound.arguments["copy"] = None

                    try:
                        return original_function(*bound.args, **bound.kwargs)
                    except ValueError as error:
                        message = str(error)
                        if "copy" in message or "Incorrect length of array annotation" in message:
                            # Retry without copy and without array annotations.
                            kwargs_retry = dict(bound.kwargs)
                            kwargs_retry.pop("copy", None)
                            if "array_annotations" in signature.parameters:
                                kwargs_retry["array_annotations"] = {}
                            return original_function(*bound.args, **kwargs_retry)
                        raise

                return patched_function

            setattr(
                analogsignal,
                function_name,
                make_patched_function(original_function, signature),
            )
            applied.append("neo.core.analogsignal." + function_name)

        # Patch AnalogSignal.__new__ as a backup.
        if hasattr(analogsignal, "AnalogSignal"):
            cls = analogsignal.AnalogSignal
            original_new = cls.__new__

            def patched_new(cls_, *args, **kwargs):
                if kwargs.get("annotations") is None:
                    kwargs["annotations"] = {}
                # Drop old / invalid array annotations; they can have the wrong length
                # for modern Neo and are not needed for metadata summaries.
                kwargs["array_annotations"] = {}
                if "copy" in kwargs:
                    kwargs["copy"] = None

                try:
                    return original_new(cls_, *args, **kwargs)
                except ValueError as error:
                    message = str(error)
                    if "copy" in message or "Incorrect length of array annotation" in message:
                        kwargs.pop("copy", None)
                        kwargs["array_annotations"] = {}
                        return original_new(cls_, *args, **kwargs)
                    raise

            cls.__new__ = staticmethod(patched_new)
            applied.append("neo.core.analogsignal.AnalogSignal.__new__")

    except Exception as error:
        if verbose:
            print("Could not patch analogsignal compatibility:", repr(error))



    # Patch Event reconstruction. Old pickles may pass a Segment object where
    # modern Neo expects an annotations mapping. For metadata extraction, links
    # to parent Segment are not needed during unpickling, so we drop invalid
    # annotations/array_annotations.
    try:
        import neo.core.event as event_module

        if hasattr(event_module, "_new_event"):
            original_function = event_module._new_event
            signature = inspect.signature(original_function)

            def patched_new_event(*args, **kwargs):
                bound = signature.bind_partial(*args, **kwargs)

                if "annotations" in signature.parameters:
                    annotations = bound.arguments.get("annotations")
                    if annotations is None or not isinstance(annotations, dict):
                        bound.arguments["annotations"] = {}

                if "array_annotations" in signature.parameters:
                    array_annotations = bound.arguments.get("array_annotations")
                    if array_annotations is None or not isinstance(array_annotations, dict):
                        bound.arguments["array_annotations"] = {}

                try:
                    return original_function(*bound.args, **bound.kwargs)
                except TypeError as error:
                    message = str(error)
                    if "argument after ** must be a mapping" in message:
                        # Retry with invalid mapping-like arguments removed.
                        retry_arguments = dict(bound.arguments)
                        if "annotations" in retry_arguments:
                            retry_arguments["annotations"] = {}
                        if "array_annotations" in retry_arguments:
                            retry_arguments["array_annotations"] = {}
                        return original_function(**retry_arguments)
                    raise
                except ValueError as error:
                    message = str(error)
                    if "Incorrect length of array annotation" in message:
                        retry_arguments = dict(bound.arguments)
                        if "array_annotations" in retry_arguments:
                            retry_arguments["array_annotations"] = {}
                        return original_function(**retry_arguments)
                    raise

            event_module._new_event = patched_new_event
            applied.append("neo.core.event._new_event")

        if hasattr(event_module, "Event"):
            cls = event_module.Event
            original_new = cls.__new__

            def patched_event_new(cls_, *args, **kwargs):
                if kwargs.get("annotations") is None or not isinstance(kwargs.get("annotations", {}), dict):
                    kwargs["annotations"] = {}
                if kwargs.get("array_annotations") is None or not isinstance(kwargs.get("array_annotations", {}), dict):
                    kwargs["array_annotations"] = {}
                return original_new(cls_, *args, **kwargs)

            cls.__new__ = staticmethod(patched_event_new)
            applied.append("neo.core.event.Event.__new__")

    except Exception as error:
        if verbose:
            print("Could not patch event compatibility:", repr(error))

    # Patch Epoch reconstruction for the same old-pickle issue.
    try:
        import neo.core.epoch as epoch_module

        for function_name in ["_new_epoch", "_new_Epoch"]:
            if not hasattr(epoch_module, function_name):
                continue

            original_function = getattr(epoch_module, function_name)
            signature = inspect.signature(original_function)

            def make_patched_epoch_function(original_function, signature):
                def patched_epoch_function(*args, **kwargs):
                    bound = signature.bind_partial(*args, **kwargs)

                    if "annotations" in signature.parameters:
                        annotations = bound.arguments.get("annotations")
                        if annotations is None or not isinstance(annotations, dict):
                            bound.arguments["annotations"] = {}

                    if "array_annotations" in signature.parameters:
                        array_annotations = bound.arguments.get("array_annotations")
                        if array_annotations is None or not isinstance(array_annotations, dict):
                            bound.arguments["array_annotations"] = {}

                    try:
                        return original_function(*bound.args, **bound.kwargs)
                    except (TypeError, ValueError) as error:
                        message = str(error)
                        if (
                            "argument after ** must be a mapping" in message
                            or "Incorrect length of array annotation" in message
                        ):
                            retry_arguments = dict(bound.arguments)
                            if "annotations" in retry_arguments:
                                retry_arguments["annotations"] = {}
                            if "array_annotations" in retry_arguments:
                                retry_arguments["array_annotations"] = {}
                            return original_function(**retry_arguments)
                        raise

                return patched_epoch_function

            setattr(epoch_module, function_name, make_patched_epoch_function(original_function, signature))
            applied.append("neo.core.epoch." + function_name)

    except Exception as error:
        if verbose:
            print("Could not patch epoch compatibility:", repr(error))

    if verbose:
        if applied:
            print("Applied compatibility patches:")
            for item in applied:
                print("  -", item)
        else:
            print("No Neo compatibility patches applied.")

    return applied


def json_safe(value, max_list_items=20, depth=0, max_depth=4):
    """
    Convert arbitrary values to JSON-safe compact values.
    Arrays and long lists are summarized, not expanded.
    """

    if depth > max_depth:
        return str(type(value).__name__)

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
            mag = value.magnitude
            try:
                import numpy as np
                if isinstance(mag, np.ndarray):
                    if mag.size == 1:
                        return {
                            "value": float(mag),
                            "unit": str(value.units),
                        }
                    return {
                        "quantity_summary": True,
                        "shape": list(mag.shape),
                        "dtype": str(mag.dtype),
                        "unit": str(value.units),
                    }
            except Exception:
                pass

            try:
                return {
                    "value": float(mag),
                    "unit": str(value.units),
                }
            except Exception:
                return str(value)

    except Exception:
        pass

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k)
            lower = key.lower()
            if any(word in lower for word in LARGE_FIELD_KEYWORDS):
                out[key] = summarize_large_value(v)
            else:
                out[key] = json_safe(
                    v,
                    max_list_items=max_list_items,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
        return out

    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if len(values) > max_list_items:
            return {
                "list_summary": True,
                "length": len(values),
                "preview": [
                    json_safe(v, max_list_items=max_list_items, depth=depth + 1, max_depth=max_depth)
                    for v in values[:5]
                ],
            }
        return [
            json_safe(v, max_list_items=max_list_items, depth=depth + 1, max_depth=max_depth)
            for v in values
        ]

    try:
        return str(value)
    except Exception:
        return None


def summarize_large_value(value):
    """
    Compact summary for fields likely to be arrays/signals/waveforms.
    """

    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return {
                "skipped_large_field": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    except Exception:
        pass

    try:
        length = len(value)
        return {
            "skipped_large_field": True,
            "type": type(value).__name__,
            "length": int(length),
        }
    except Exception:
        return {
            "skipped_large_field": True,
            "type": type(value).__name__,
        }


def parse_touchandsee_filename(path):
    """
    Extract subject, sample id, date from filenames like:
    hbp-01681_TouchAndSee_Ramachandran_samp20__2012-08-09.pkl
    """

    name = Path(path).name

    pattern = (
        r"hbp-01681_TouchAndSee_"
        r"(?P<subject>[^_]+)_"
        r"(?P<sample>samp\d+)__"
        r"(?P<date>\d{4}-\d{2}-\d{2})\.pkl$"
    )

    match = re.match(pattern, name)

    if match:
        subject = match.group("subject")
        sample = match.group("sample")
        date = match.group("date")
        session_id = f"{subject}_{sample}_{date}"
        return {
            "subject_id": subject,
            "animal_id": subject,
            "sample_id": sample,
            "session_date": date,
            "session_id": session_id,
            "dataset_code": "hbp-01681",
            "project_label": "TouchAndSee",
        }

    return {
        "subject_id": None,
        "animal_id": None,
        "sample_id": None,
        "session_date": None,
        "session_id": Path(path).stem,
        "dataset_code": "hbp-01681",
        "project_label": "TouchAndSee",
    }


def summarize_annotations(obj, max_keys=50):
    """
    Return annotation keys and compact values for non-large annotations.
    """

    annotations = getattr(obj, "annotations", None)

    if not annotations:
        return {
            "annotation_keys": [],
            "annotations_compact": {},
        }

    keys = [str(k) for k in annotations.keys()]
    compact = {}

    for key in keys[:max_keys]:
        lower = key.lower()
        if any(word in lower for word in LARGE_FIELD_KEYWORDS):
            compact[key] = summarize_large_value(annotations.get(key))
        else:
            compact[key] = json_safe(annotations.get(key))

    return {
        "annotation_keys": keys,
        "annotations_compact": compact,
        "annotations_truncated": len(keys) > max_keys,
    }


def summarize_spiketrain(st):
    """
    Compact summary of one Neo SpikeTrain.
    No spike times are stored.
    """

    out = {
        "name": json_safe(getattr(st, "name", None)),
        "description": json_safe(getattr(st, "description", None)),
        "n_spikes": None,
        "t_start": None,
        "t_stop": None,
        "duration": None,
        "units": None,
        "sampling_rate": None,
    }

    try:
        out["n_spikes"] = int(len(st))
    except Exception:
        pass

    for attr in ["t_start", "t_stop", "duration", "sampling_rate"]:
        try:
            out[attr] = json_safe(getattr(st, attr))
        except Exception:
            pass

    try:
        out["units"] = str(getattr(st, "units", None))
    except Exception:
        pass

    out.update(summarize_annotations(st))

    return out


def summarize_analogsignal(sig):
    """
    Compact summary of one Neo AnalogSignal.
    No signal values are stored.
    """

    out = {
        "name": json_safe(getattr(sig, "name", None)),
        "description": json_safe(getattr(sig, "description", None)),
        "shape": None,
        "n_samples": None,
        "n_channels": None,
        "units": None,
        "sampling_rate": None,
        "t_start": None,
        "duration": None,
    }

    try:
        shape = list(sig.shape)
        out["shape"] = shape
        out["n_samples"] = int(shape[0]) if len(shape) > 0 else None
        out["n_channels"] = int(shape[1]) if len(shape) > 1 else 1
    except Exception:
        pass

    try:
        out["units"] = str(getattr(sig, "units", None))
    except Exception:
        pass

    for attr in ["sampling_rate", "t_start", "duration"]:
        try:
            out[attr] = json_safe(getattr(sig, attr))
        except Exception:
            pass

    out.update(summarize_annotations(sig))

    return out


def summarize_event(ev, max_labels=20):
    """
    Compact summary of Neo Event.
    Does not store event times.
    """

    out = {
        "name": json_safe(getattr(ev, "name", None)),
        "description": json_safe(getattr(ev, "description", None)),
        "n_events": None,
        "labels_preview": [],
    }

    try:
        out["n_events"] = int(len(ev))
    except Exception:
        pass

    try:
        labels = getattr(ev, "labels", [])
        out["labels_preview"] = [str(x) for x in list(labels[:max_labels])]
    except Exception:
        pass

    out.update(summarize_annotations(ev))

    return out


def summarize_epoch(ep, max_labels=20):
    """
    Compact summary of Neo Epoch.
    Does not store epoch times.
    """

    out = {
        "name": json_safe(getattr(ep, "name", None)),
        "description": json_safe(getattr(ep, "description", None)),
        "n_epochs": None,
        "labels_preview": [],
    }

    try:
        out["n_epochs"] = int(len(ep))
    except Exception:
        pass

    try:
        labels = getattr(ep, "labels", [])
        out["labels_preview"] = [str(x) for x in list(labels[:max_labels])]
    except Exception:
        pass

    out.update(summarize_annotations(ep))

    return out


def summarize_unit(unit):
    """
    Compact summary of one Neo Unit, if Block has unit/channel hierarchy.
    """

    out = {
        "name": json_safe(getattr(unit, "name", None)),
        "description": json_safe(getattr(unit, "description", None)),
        "n_spiketrains": None,
        "n_spikes_total": None,
    }

    try:
        sts = getattr(unit, "spiketrains", [])
        out["n_spiketrains"] = int(len(sts))
        out["n_spikes_total"] = int(sum(len(st) for st in sts))
    except Exception:
        pass

    out.update(summarize_annotations(unit))

    return out


def summarize_neo_block(block, include_object_details=True):
    """
    Compact summary of a Neo Block.
    """

    out = {
        "object_type": type(block).__name__,
        "name": json_safe(getattr(block, "name", None)),
        "description": json_safe(getattr(block, "description", None)),
        "n_segments": 0,
        "n_channel_indexes": 0,
        "n_units": 0,
        "n_spiketrains": 0,
        "n_spikes_total": 0,
        "n_analogsignals": 0,
        "n_events": 0,
        "n_event_times_total": 0,
        "n_epochs": 0,
        "n_epoch_times_total": 0,
        "segment_summaries": [],
        "unit_summaries": [],
    }

    out.update(summarize_annotations(block))

    segments = getattr(block, "segments", []) or []
    out["n_segments"] = int(len(segments))

    try:
        channel_indexes = getattr(block, "channel_indexes", []) or []
        out["n_channel_indexes"] = int(len(channel_indexes))
    except Exception:
        pass

    # Units can be stored in different locations depending on Neo version.
    units = []
    try:
        units = list(getattr(block, "list_units", []) or [])
    except Exception:
        units = []

    if not units:
        try:
            for chx in getattr(block, "channel_indexes", []) or []:
                units.extend(list(getattr(chx, "units", []) or []))
        except Exception:
            pass

    out["n_units"] = int(len(units))

    if include_object_details:
        out["unit_summaries"] = [summarize_unit(unit) for unit in units[:200]]
        out["unit_summaries_truncated"] = len(units) > 200

    for segment in segments:
        seg_out = {
            "name": json_safe(getattr(segment, "name", None)),
            "description": json_safe(getattr(segment, "description", None)),
            "n_spiketrains": int(len(getattr(segment, "spiketrains", []) or [])),
            "n_analogsignals": int(len(getattr(segment, "analogsignals", []) or [])),
            "n_events": int(len(getattr(segment, "events", []) or [])),
            "n_epochs": int(len(getattr(segment, "epochs", []) or [])),
            "spiketrain_summaries": [],
            "analogsignal_summaries": [],
            "event_summaries": [],
            "epoch_summaries": [],
        }

        seg_out.update(summarize_annotations(segment))

        spike_total = 0
        for st in getattr(segment, "spiketrains", []) or []:
            try:
                spike_total += int(len(st))
            except Exception:
                pass

        event_total = 0
        for ev in getattr(segment, "events", []) or []:
            try:
                event_total += int(len(ev))
            except Exception:
                pass

        epoch_total = 0
        for ep in getattr(segment, "epochs", []) or []:
            try:
                epoch_total += int(len(ep))
            except Exception:
                pass

        out["n_spiketrains"] += seg_out["n_spiketrains"]
        out["n_spikes_total"] += spike_total
        out["n_analogsignals"] += seg_out["n_analogsignals"]
        out["n_events"] += seg_out["n_events"]
        out["n_event_times_total"] += event_total
        out["n_epochs"] += seg_out["n_epochs"]
        out["n_epoch_times_total"] += epoch_total

        seg_out["n_spikes_total"] = spike_total
        seg_out["n_event_times_total"] = event_total
        seg_out["n_epoch_times_total"] = epoch_total

        if include_object_details:
            seg_out["spiketrain_summaries"] = [
                summarize_spiketrain(st) for st in (getattr(segment, "spiketrains", []) or [])[:200]
            ]
            seg_out["spiketrain_summaries_truncated"] = seg_out["n_spiketrains"] > 200

            seg_out["analogsignal_summaries"] = [
                summarize_analogsignal(sig) for sig in (getattr(segment, "analogsignals", []) or [])[:50]
            ]
            seg_out["analogsignal_summaries_truncated"] = seg_out["n_analogsignals"] > 50

            seg_out["event_summaries"] = [
                summarize_event(ev) for ev in (getattr(segment, "events", []) or [])[:100]
            ]
            seg_out["event_summaries_truncated"] = seg_out["n_events"] > 100

            seg_out["epoch_summaries"] = [
                summarize_epoch(ep) for ep in (getattr(segment, "epochs", []) or [])[:100]
            ]
            seg_out["epoch_summaries_truncated"] = seg_out["n_epochs"] > 100

        out["segment_summaries"].append(seg_out)

    out["has_spike_metadata"] = out["n_spiketrains"] > 0 or out["n_units"] > 0
    out["has_unit_metadata"] = out["n_units"] > 0
    out["has_lfp_metadata"] = out["n_analogsignals"] > 0
    out["has_event_metadata"] = out["n_events"] > 0
    out["has_epoch_metadata"] = out["n_epochs"] > 0

    return json_safe(out)


def load_pickle(path):
    """
    Load one pickle after compatibility patch.
    """

    patch_old_neo_pickle_compatibility(verbose=False)
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_one_file(path, dataset_root, include_object_details=True):
    """
    Extract internal metadata from one TouchAndSee pickle.
    """

    path = Path(path)
    dataset_root = Path(dataset_root)

    file_info = parse_touchandsee_filename(path)

    result = {
        "file_metadata": {
            "path": str(path.relative_to(dataset_root)) if dataset_root in path.parents else str(path),
            "absolute_path": str(path),
            "file_name": path.name,
            "file_extension": path.suffix,
            "file_size_bytes": int(path.stat().st_size),
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
            **file_info,
        },
        "touchandsee_extraction": {
            "attempted": True,
            "success": False,
            "error": None,
            "traceback": None,
            "object_type": None,
            "object_summary": None,
        },
    }

    try:
        obj = load_pickle(path)
        result["touchandsee_extraction"]["success"] = True
        result["touchandsee_extraction"]["object_type"] = type(obj).__name__

        # Most files are expected to be Neo Blocks.
        if hasattr(obj, "segments"):
            result["touchandsee_extraction"]["object_summary"] = summarize_neo_block(
                obj,
                include_object_details=include_object_details,
            )
        else:
            result["touchandsee_extraction"]["object_summary"] = {
                "object_type": type(obj).__name__,
                "repr": str(obj)[:500],
            }

    except Exception as error:
        result["touchandsee_extraction"]["success"] = False
        result["touchandsee_extraction"]["error"] = repr(error)
        result["touchandsee_extraction"]["traceback"] = traceback.format_exc()

    return result


def build_dataset_summary(file_results):
    """
    Dataset-level summary across files.
    """

    pkl_results = [
        item for item in file_results
        if item["file_metadata"].get("file_extension") == ".pkl"
    ]

    successes = [
        item for item in pkl_results
        if item["touchandsee_extraction"].get("success") is True
    ]

    failures = [
        item for item in pkl_results
        if item["touchandsee_extraction"].get("success") is False
    ]

    subjects = sorted(
        list(
            set(
                item["file_metadata"].get("subject_id")
                for item in pkl_results
                if item["file_metadata"].get("subject_id")
            )
        )
    )

    sessions = sorted(
        list(
            set(
                item["file_metadata"].get("session_id")
                for item in pkl_results
                if item["file_metadata"].get("session_id")
            )
        )
    )

    totals = {
        "n_spiketrains": 0,
        "n_spikes_total": 0,
        "n_units": 0,
        "n_analogsignals": 0,
        "n_events": 0,
        "n_event_times_total": 0,
        "n_epochs": 0,
        "n_epoch_times_total": 0,
    }

    sessions_with_spikes = 0
    sessions_with_lfp = 0
    sessions_with_events = 0

    for item in successes:
        summary = (
            item.get("touchandsee_extraction", {})
            .get("object_summary", {})
        )

        for key in totals:
            totals[key] += int(summary.get(key) or 0)

        if summary.get("has_spike_metadata"):
            sessions_with_spikes += 1
        if summary.get("has_lfp_metadata"):
            sessions_with_lfp += 1
        if summary.get("has_event_metadata"):
            sessions_with_events += 1

    return {
        "n_files": int(len(file_results)),
        "n_pickle_files": int(len(pkl_results)),
        "n_pickle_success": int(len(successes)),
        "n_pickle_failure": int(len(failures)),
        "n_subjects_detected": int(len(subjects)),
        "subjects_detected": subjects,
        "n_sessions_detected": int(len(sessions)),
        "sessions_detected": sessions,
        **totals,
        "n_sessions_with_spike_metadata": int(sessions_with_spikes),
        "n_sessions_with_lfp_metadata": int(sessions_with_lfp),
        "n_sessions_with_event_metadata": int(sessions_with_events),
    }


def extract_touchandsee_dataset(dataset_path, output_folder=None, include_object_details=True, test_one=False):
    """
    Extract internal metadata from all TouchAndSee pickle files.
    """

    dataset_path = Path(dataset_path)

    if output_folder is None:
        output_folder = Path(os.getcwd()) / "outputs" / "extracted_metadata"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    patch_old_neo_pickle_compatibility(verbose=True)

    pkl_files = sorted(dataset_path.rglob("*.pkl"))

    if test_one:
        pkl_files = pkl_files[:1]

    print("Dataset path:", dataset_path)
    print("Pickle files found:", len(pkl_files))

    results = []

    for path in pkl_files:
        print("Extracting TouchAndSee:", path.name)
        result = extract_one_file(
            path=path,
            dataset_root=dataset_path,
            include_object_details=include_object_details,
        )
        results.append(result)

        if result["touchandsee_extraction"]["success"] is False:
            print("  FAILED:", result["touchandsee_extraction"]["error"])
        else:
            summary = result["touchandsee_extraction"]["object_summary"] or {}
            print(
                "  OK:",
                "spiketrains=", summary.get("n_spiketrains"),
                "spikes=", summary.get("n_spikes_total"),
                "analogsignals=", summary.get("n_analogsignals"),
                "events=", summary.get("n_events"),
            )

    dataset_metadata = {
        "dataset_name": dataset_path.name,
        "dataset_folder": str(dataset_path),
        "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extractor": "touchandsee_internal_neo_pickle_extractor_v3",
        "include_object_details": bool(include_object_details),
        "dataset_summary": build_dataset_summary(results),
        "files": results,
    }

    suffix = "_touchandsee_internal_metadata_TEST.json" if test_one else "_touchandsee_internal_metadata.json"
    output_json = output_folder / (dataset_path.name + suffix)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)

    print()
    print("TouchAndSee extraction finished")
    print("Output:", output_json)
    print("Dataset summary:")
    print(json.dumps(dataset_metadata["dataset_summary"], indent=2))

    return dataset_metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract internal metadata from TouchAndSee Neo pickle dataset."
    )

    parser.add_argument(
        "dataset_path",
        help="Path to p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681 dataset folder",
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        help="Output folder. Default: outputs/extracted_metadata",
    )

    parser.add_argument(
        "--no_object_details",
        action="store_true",
        help="Only save counts/summaries, no per-spiketrain/event summaries.",
    )

    parser.add_argument(
        "--test_one",
        action="store_true",
        help="Only test the first pickle file.",
    )

    args = parser.parse_args()

    extract_touchandsee_dataset(
        dataset_path=args.dataset_path,
        output_folder=args.output_folder,
        include_object_details=not args.no_object_details,
        test_one=args.test_one,
    )
