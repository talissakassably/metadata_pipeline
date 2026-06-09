# -*- coding: utf-8 -*-
"""
TouchAndSee internal metadata extractor for old Neo pickle files.

Dataset example:
    p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681

Why this script exists:
    These .pkl files were saved with an old Neo version. Modern Neo may fail
    while unpickling old objects. This script patches the common old-Neo
    incompatibilities, then extracts only compact metadata.

It DOES extract:
    - subject / session identity from path and filename
    - Block / Segment counts
    - SpikeTrain counts and total spikes
    - AnalogSignal counts, shapes, units, sampling rates
    - Event and Epoch counts / names
    - annotation keys and compact annotation previews

It does NOT save:
    - raw signal values
    - full spike times
    - waveform arrays
    - large numpy arrays
"""

import os
import re
import sys
import json
import pickle
import types
import argparse
import traceback
import inspect
from pathlib import Path
from datetime import datetime


EXTRACTOR_NAME = "touchandsee_internal_neo_pickle_extractor_v14"

LARGE_FIELD_KEYWORDS = [
    "waveform",
    "waveforms",
    "signal",
    "signals",
    "spike_times",
    "spiketimes",
    "times",
    "data",
    "array",
    "template",
    "templates",
]


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def is_large_key(key):
    key = str(key).lower()
    return any(word in key for word in LARGE_FIELD_KEYWORDS)


def make_json_safe(value, max_list=20, max_dict=40):
    """Convert values to compact JSON-safe summaries."""

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
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
            magnitude = value.magnitude
            try:
                import numpy as np
                if isinstance(magnitude, np.ndarray):
                    if magnitude.size == 1:
                        return {
                            "value": float(magnitude.reshape(-1)[0]),
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
            try:
                return {"value": float(magnitude), "unit": str(value.units)}
            except Exception:
                return str(value)
    except Exception:
        pass

    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= max_dict:
                out["__truncated__"] = True
                out["__original_length__"] = len(value)
                break
            if is_large_key(k):
                out[str(k)] = summarize_large_value(v)
            else:
                out[str(k)] = make_json_safe(v, max_list=max_list, max_dict=max_dict)
        return out

    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if len(values) > max_list:
            return {
                "list_summary": True,
                "length": len(values),
                "preview": [make_json_safe(x) for x in values[:5]],
            }
        return [make_json_safe(x) for x in values]

    try:
        return str(value)
    except Exception:
        return None


def summarize_large_value(value):
    """Do not expand raw arrays or large fields."""
    try:
        return make_json_safe(value)
    except Exception:
        return "SKIPPED_LARGE_FIELD"


def annotation_summary(obj):
    annotations = getattr(obj, "annotations", None)
    if not isinstance(annotations, dict):
        return {"annotation_keys": []}

    keys = sorted([str(k) for k in annotations.keys()])
    preview = {}
    for key in keys[:30]:
        value = annotations.get(key)
        if is_large_key(key):
            preview[key] = "SKIPPED_LARGE_FIELD"
        else:
            preview[key] = make_json_safe(value)

    return {
        "annotation_keys": keys,
        "annotation_preview": preview,
    }


# ---------------------------------------------------------------------
# Old Neo compatibility patches
# ---------------------------------------------------------------------

def _safe_len(value):
    try:
        return len(value)
    except Exception:
        return None


def _is_mapping(value):
    return isinstance(value, dict)


def _as_numeric_array(value):
    try:
        import numpy as np
        if hasattr(value, "magnitude"):
            value = value.magnitude
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _max_time_value(times):
    arr = _as_numeric_array(times)
    if arr is None:
        return None
    try:
        return float(arr.max())
    except Exception:
        return None


def _min_time_value(times):
    arr = _as_numeric_array(times)
    if arr is None:
        return None
    try:
        return float(arr.min())
    except Exception:
        return None


def _time_with_units(value, reference_times=None, unit_text="s"):
    """Return a scalar time compatible with Neo constructors."""
    if hasattr(value, "units"):
        return value

    if reference_times is not None and hasattr(reference_times, "units"):
        try:
            return float(value) * reference_times.units
        except Exception:
            pass

    return float(value)


def _sanitize_time_value(value, default, reference_times=None):
    if value is None:
        return _time_with_units(default, reference_times)

    if isinstance(value, (int, float)):
        return _time_with_units(value, reference_times)

    if hasattr(value, "units"):
        return value

    try:
        import numpy as np
        if isinstance(value, np.generic):
            return _time_with_units(float(value), reference_times)
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return _time_with_units(float(value.reshape(-1)[0]), reference_times)
            return _time_with_units(default, reference_times)
    except Exception:
        pass

    # Old pickle can put dict/Segment/object here. Ignore and use fallback.
    return _time_with_units(default, reference_times)


def _ensure_units(arguments):
    if "times" in arguments:
        times = arguments.get("times")
        if times is not None and not hasattr(times, "units"):
            if arguments.get("units") is None:
                arguments["units"] = "s"
    return arguments


def _fix_labels(arguments):
    if "labels" not in arguments or "times" not in arguments:
        return arguments
    n_times = _safe_len(arguments.get("times"))
    n_labels = _safe_len(arguments.get("labels"))
    if n_times is not None and n_labels != n_times:
        arguments["labels"] = [""] * int(n_times)
    return arguments


def _sanitize_mapping_arguments(arguments):
    for key in ["annotations", "array_annotations"]:
        if key in arguments and not _is_mapping(arguments.get(key)):
            arguments[key] = {}
        if key in arguments and arguments.get(key) is None:
            arguments[key] = {}
    return arguments


def _sanitize_waveforms(arguments):
    if "waveforms" in arguments:
        wf = arguments.get("waveforms")
        if wf is not None and not hasattr(wf, "shape"):
            arguments["waveforms"] = None
    return arguments


def _sanitize_spiketrain_times(arguments):
    times = arguments.get("times")
    max_t = _max_time_value(times)
    min_t = _min_time_value(times)

    if "t_start" in arguments:
        default_start = 0.0 if min_t is None else min(0.0, min_t)
        arguments["t_start"] = _sanitize_time_value(arguments.get("t_start"), default_start, times)

    if "t_stop" in arguments:
        current_stop = arguments.get("t_stop")
        stop_bad = False

        if current_stop is None:
            stop_bad = True
        elif isinstance(current_stop, (dict, list, tuple)):
            stop_bad = True
        else:
            current_stop_numeric = _max_time_value(current_stop)
            if current_stop_numeric is None:
                try:
                    current_stop_numeric = float(current_stop)
                except Exception:
                    current_stop_numeric = None
            if max_t is not None and current_stop_numeric is not None and current_stop_numeric < max_t:
                stop_bad = True

        if stop_bad:
            fallback_stop = 1.0 if max_t is None else max_t + 1e-6
            arguments["t_stop"] = _time_with_units(fallback_stop, times)
        else:
            arguments["t_stop"] = _sanitize_time_value(current_stop, 1.0 if max_t is None else max_t + 1e-6, times)

    return arguments


def install_old_neo_pickle_patches(verbose=True):
    """Install monkey patches so old Neo pickles can be loaded enough for metadata extraction."""
    applied = []

    # Lightweight shims for classes/modules removed from modern Neo.
    try:
        class LegacyNeoContainer(object):
            def __init__(self, *args, **kwargs):
                self._legacy_args = args
                self.__dict__.update(kwargs)
                if not hasattr(self, "name"):
                    self.name = None
                if not hasattr(self, "annotations") or self.annotations is None:
                    self.annotations = {}
                if not hasattr(self, "spiketrains"):
                    self.spiketrains = []
                if not hasattr(self, "analogsignals"):
                    self.analogsignals = []

            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)
                elif isinstance(state, tuple):
                    for item in state:
                        if isinstance(item, dict):
                            self.__dict__.update(item)
                if not hasattr(self, "annotations") or self.annotations is None:
                    self.annotations = {}
                if not hasattr(self, "spiketrains"):
                    self.spiketrains = []
                if not hasattr(self, "analogsignals"):
                    self.analogsignals = []

        legacy_modules = {
            "neo.core.unit": "Unit",
            "neo.core.channelindex": "ChannelIndex",
            "neo.core.recordingchannelgroup": "RecordingChannelGroup",
            "neo.core.recordingchannel": "RecordingChannel",
        }

        for module_name, class_name in legacy_modules.items():
            if module_name not in sys.modules:
                module = types.ModuleType(module_name)
                LegacyClass = type(class_name, (LegacyNeoContainer,), {"__module__": module_name})
                setattr(module, class_name, LegacyClass)
                sys.modules[module_name] = module
                applied.append(module_name + "." + class_name + " shim")
    except Exception as error:
        if verbose:
            print("Could not install old Neo shims:", repr(error))

    # Fallback object used when an old SpikeTrain cannot be reconstructed by Neo.
    class LegacySpikeTrainFallback(object):
        def __init__(self, times=None, units="s", t_start=None, t_stop=None, name=None, file_origin=None, description=None, annotations=None, **kwargs):
            self.times = times
            self.units = units
            self.t_start = t_start
            self.t_stop = t_stop
            self.name = name
            self.file_origin = file_origin
            self.description = description
            self.annotations = annotations if isinstance(annotations, dict) else {}
            for k, v in kwargs.items():
                if k not in ["waveforms", "array_annotations"]:
                    setattr(self, k, v)

        def __len__(self):
            try:
                return len(self.times)
            except Exception:
                return 0

    # Patch AnalogSignal reconstruction.
    try:
        import neo.core.analogsignal as analogsignal_module
        for function_name in ["_new_AnalogSignalArray", "_new_AnalogSignal"]:
            if not hasattr(analogsignal_module, function_name):
                continue
            original = getattr(analogsignal_module, function_name)
            signature = inspect.signature(original)

            def make_patch(original, signature):
                def patched(*args, **kwargs):
                    bound = signature.bind_partial(*args, **kwargs)
                    _sanitize_mapping_arguments(bound.arguments)
                    if "array_annotations" in bound.arguments:
                        bound.arguments["array_annotations"] = {}
                    if "copy" in bound.arguments:
                        bound.arguments["copy"] = None
                    try:
                        return original(*bound.args, **bound.kwargs)
                    except Exception:
                        retry = dict(bound.arguments)
                        _sanitize_mapping_arguments(retry)
                        if "array_annotations" in retry:
                            retry["array_annotations"] = {}
                        retry.pop("copy", None)
                        return original(**retry)
                return patched

            setattr(analogsignal_module, function_name, make_patch(original, signature))
            applied.append("neo.core.analogsignal." + function_name)
    except Exception as error:
        if verbose:
            print("Could not patch AnalogSignal:", repr(error))

    # Patch Event reconstruction.
    try:
        import neo.core.event as event_module
        if hasattr(event_module, "_new_event"):
            original = event_module._new_event
            signature = inspect.signature(original)

            def patched_new_event(*args, _original=original, _signature=signature, **kwargs):
                # Important: bind original/signature as defaults so later patches
                # do not overwrite this closure.
                bound = _signature.bind_partial(*args, **kwargs)
                _sanitize_mapping_arguments(bound.arguments)
                _fix_labels(bound.arguments)
                _ensure_units(bound.arguments)
                try:
                    return _original(*bound.args, **bound.kwargs)
                except Exception:
                    retry = dict(bound.arguments)
                    _sanitize_mapping_arguments(retry)
                    _fix_labels(retry)
                    _ensure_units(retry)
                    return _original(**retry)

            event_module._new_event = patched_new_event
            applied.append("neo.core.event._new_event")
    except Exception as error:
        if verbose:
            print("Could not patch Event:", repr(error))

    # Patch Epoch reconstruction.
    try:
        import neo.core.epoch as epoch_module
        if hasattr(epoch_module, "_new_epoch"):
            original = epoch_module._new_epoch
            signature = inspect.signature(original)

            def patched_new_epoch(*args, _original=original, _signature=signature, **kwargs):
                # Important: bind original/signature as defaults so later patches
                # do not overwrite this closure.
                bound = _signature.bind_partial(*args, **kwargs)
                _sanitize_mapping_arguments(bound.arguments)
                _fix_labels(bound.arguments)
                _ensure_units(bound.arguments)
                try:
                    return _original(*bound.args, **bound.kwargs)
                except Exception:
                    retry = dict(bound.arguments)
                    _sanitize_mapping_arguments(retry)
                    _fix_labels(retry)
                    _ensure_units(retry)
                    if retry.get("units") is None:
                        retry["units"] = "s"
                    return _original(**retry)

            epoch_module._new_epoch = patched_new_epoch
            applied.append("neo.core.epoch._new_epoch")
    except Exception as error:
        if verbose:
            print("Could not patch Epoch:", repr(error))

    # Patch SpikeTrain reconstruction.
    try:
        import neo.core.spiketrain as st_module

        if hasattr(st_module, "_new_spiketrain"):
            original = st_module._new_spiketrain
            signature = inspect.signature(original)

            def _legacy_spiketrain_from_arguments(arguments):
                times = arguments.get("times")
                if times is None:
                    # Some old pickle layouts can put times in a positional-like field.
                    for key in ["signal", "data", "spike_times"]:
                        if key in arguments:
                            times = arguments.get(key)
                            break

                max_t = _max_time_value(times)
                t_start = arguments.get("t_start")
                t_stop = arguments.get("t_stop")

                if t_start is None or isinstance(t_start, (dict, list, tuple)):
                    t_start = 0.0
                if t_stop is None or isinstance(t_stop, (dict, list, tuple)):
                    t_stop = 1.0 if max_t is None else max_t + 1e-6

                annotations = arguments.get("annotations")
                if not isinstance(annotations, dict):
                    annotations = {}

                return LegacySpikeTrainFallback(
                    times=times,
                    units=arguments.get("units") or "s",
                    t_start=t_start,
                    t_stop=t_stop,
                    name=arguments.get("name"),
                    file_origin=arguments.get("file_origin"),
                    description=arguments.get("description"),
                    annotations=annotations,
                )

            def patched_new_spiketrain(*args, **kwargs):
                """
                Very old Neo pickles sometimes pass Segment objects in fields where
                newer Neo expects waveforms/annotations. For this metadata extractor,
                we do not need to reconstruct full Neo SpikeTrain objects. We only need
                len(times), so always return a lightweight fallback object.
                """
                try:
                    bound = signature.bind_partial(*args, **kwargs)
                    arguments = dict(bound.arguments)
                except Exception:
                    arguments = dict(kwargs)

                    # Old Neo _new_spiketrain commonly stores times as the second
                    # positional argument: (cls, times, ...). Keep only what is safe.
                    if len(args) >= 2:
                        arguments.setdefault("times", args[1])
                    elif len(args) >= 1:
                        arguments.setdefault("times", args[0])

                    # Try to capture optional timing fields from positional args.
                    if len(args) >= 3:
                        arguments.setdefault("t_stop", args[2])

                _sanitize_mapping_arguments(arguments)
                _sanitize_waveforms(arguments)
                _ensure_units(arguments)
                _sanitize_spiketrain_times(arguments)

                return _legacy_spiketrain_from_arguments(arguments)

            st_module._new_spiketrain = patched_new_spiketrain
            applied.append("neo.core.spiketrain._new_spiketrain with fallback")
    except Exception as error:
        if verbose:
            print("Could not patch SpikeTrain:", repr(error))

    if verbose:
        print("Neo compatibility patches applied:")
        for item in applied:
            print("  -", item)


# ---------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------

def parse_touchandsee_file_metadata(file_path, dataset_path):
    file_path = Path(file_path)
    dataset_path = Path(dataset_path)

    try:
        relative_path = str(file_path.relative_to(dataset_path))
    except Exception:
        relative_path = str(file_path)

    file_name = file_path.name
    parts = file_path.parts

    subject_id = None
    sample_id = None
    session_date = None
    dataset_code = None
    project_label = None

    # Expected name:
    # hbp-01681_TouchAndSee_Ramachandran_samp20__2012-08-09.pkl
    match = re.search(
        r"(?P<dataset_code>hbp-[0-9]+)_(?P<project>[^_]+)_(?P<subject>[^_]+)_(?P<sample>samp[0-9]+)__?(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})",
        file_name,
    )

    if match:
        dataset_code = match.group("dataset_code")
        project_label = match.group("project")
        subject_id = match.group("subject")
        sample_id = match.group("sample")
        session_date = match.group("date")
    else:
        # Fallback from folder structure: TouchAndSee\Subject\sampXX\file.pkl
        for part in parts:
            if re.match(r"samp[0-9]+", part):
                sample_id = part
        if sample_id in parts:
            idx = list(parts).index(sample_id)
            if idx > 0:
                subject_id = parts[idx - 1]

    session_id = None
    if subject_id and sample_id and session_date:
        session_id = f"{subject_id}_{sample_id}_{session_date}"

    return {
        "path": relative_path,
        "absolute_path": str(file_path),
        "file_name": file_name,
        "file_extension": file_path.suffix.lower(),
        "file_size_bytes": int(file_path.stat().st_size),
        "file_size_mb": round(file_path.stat().st_size / 1024 / 1024, 3),
        "subject_id": subject_id,
        "animal_id": subject_id,
        "sample_id": sample_id,
        "session_date": session_date,
        "session_id": session_id,
        "dataset_code": dataset_code,
        "project_label": project_label,
    }


def find_pickle_files(dataset_path):
    dataset_path = Path(dataset_path)
    return sorted(dataset_path.rglob("*.pkl"))


# ---------------------------------------------------------------------
# Neo object summarizers
# ---------------------------------------------------------------------

def count_items(obj, attr_name):
    try:
        return len(getattr(obj, attr_name, []))
    except Exception:
        return 0


def summarize_spiketrain(st):
    info = {
        "name": make_json_safe(getattr(st, "name", None)),
        "n_spikes": 0,
        "t_start": None,
        "t_stop": None,
        "units": None,
        "sampling_rate": None,
    }

    try:
        info["n_spikes"] = int(len(st))
    except Exception:
        pass

    for attr in ["t_start", "t_stop", "units", "sampling_rate"]:
        try:
            info[attr] = make_json_safe(getattr(st, attr, None))
        except Exception:
            pass

    info.update(annotation_summary(st))
    return info


def summarize_analogsignal(signal):
    info = {
        "name": make_json_safe(getattr(signal, "name", None)),
        "shape": None,
        "n_samples": None,
        "n_channels": None,
        "units": None,
        "sampling_rate": None,
        "t_start": None,
        "duration": None,
    }

    try:
        shape = list(signal.shape)
        info["shape"] = shape
        if len(shape) >= 1:
            info["n_samples"] = int(shape[0])
        if len(shape) >= 2:
            info["n_channels"] = int(shape[1])
        else:
            info["n_channels"] = 1
    except Exception:
        pass

    for attr in ["units", "sampling_rate", "t_start", "duration"]:
        try:
            info[attr] = make_json_safe(getattr(signal, attr, None))
        except Exception:
            pass

    info.update(annotation_summary(signal))
    return info


def summarize_event(event):
    info = {
        "name": make_json_safe(getattr(event, "name", None)),
        "n_times": 0,
        "units": None,
        "labels_preview": [],
    }

    try:
        info["n_times"] = int(len(event))
    except Exception:
        pass

    try:
        info["units"] = make_json_safe(getattr(event, "units", None))
    except Exception:
        pass

    try:
        labels = getattr(event, "labels", None)
        if labels is not None:
            labels = list(labels)
            info["labels_preview"] = [make_json_safe(x) for x in labels[:10]]
    except Exception:
        pass

    info.update(annotation_summary(event))
    return info


def summarize_epoch(epoch):
    info = {
        "name": make_json_safe(getattr(epoch, "name", None)),
        "n_epochs": 0,
        "units": None,
        "labels_preview": [],
    }

    try:
        info["n_epochs"] = int(len(epoch))
    except Exception:
        pass

    try:
        info["units"] = make_json_safe(getattr(epoch, "units", None))
    except Exception:
        pass

    try:
        labels = getattr(epoch, "labels", None)
        if labels is not None:
            labels = list(labels)
            info["labels_preview"] = [make_json_safe(x) for x in labels[:10]]
    except Exception:
        pass

    info.update(annotation_summary(epoch))
    return info


def summarize_segment(segment, include_object_details=True):
    spiketrains = list(getattr(segment, "spiketrains", []) or [])
    analogsignals = list(getattr(segment, "analogsignals", []) or [])
    events = list(getattr(segment, "events", []) or [])
    epochs = list(getattr(segment, "epochs", []) or [])

    spike_counts = []
    for st in spiketrains:
        try:
            spike_counts.append(int(len(st)))
        except Exception:
            pass

    event_counts = {}
    for ev in events:
        name = getattr(ev, "name", None) or "unnamed_event"
        name = str(name)
        try:
            event_counts[name] = event_counts.get(name, 0) + int(len(ev))
        except Exception:
            event_counts[name] = event_counts.get(name, 0)

    epoch_counts = {}
    for ep in epochs:
        name = getattr(ep, "name", None) or "unnamed_epoch"
        name = str(name)
        try:
            epoch_counts[name] = epoch_counts.get(name, 0) + int(len(ep))
        except Exception:
            epoch_counts[name] = epoch_counts.get(name, 0)

    summary = {
        "name": make_json_safe(getattr(segment, "name", None)),
        "n_spiketrains": int(len(spiketrains)),
        "n_spikes_total": int(sum(spike_counts)),
        "min_spikes_per_spiketrain": int(min(spike_counts)) if spike_counts else None,
        "max_spikes_per_spiketrain": int(max(spike_counts)) if spike_counts else None,
        "mean_spikes_per_spiketrain": float(sum(spike_counts) / len(spike_counts)) if spike_counts else None,
        "n_analogsignals": int(len(analogsignals)),
        "n_events": int(len(events)),
        "n_event_times_total": int(sum(event_counts.values())),
        "event_names": sorted(list(event_counts.keys())),
        "event_counts": event_counts,
        "n_epochs": int(len(epochs)),
        "n_epoch_times_total": int(sum(epoch_counts.values())),
        "epoch_names": sorted(list(epoch_counts.keys())),
        "epoch_counts": epoch_counts,
    }

    summary.update(annotation_summary(segment))

    if include_object_details:
        summary["spiketrains_preview"] = [summarize_spiketrain(st) for st in spiketrains[:10]]
        summary["analogsignals"] = [summarize_analogsignal(sig) for sig in analogsignals]
        summary["events"] = [summarize_event(ev) for ev in events]
        summary["epochs"] = [summarize_epoch(ep) for ep in epochs]

    return summary


def summarize_loaded_object(obj, include_object_details=True):
    object_type = type(obj).__name__

    # Usually this should be a Neo Block.
    segments = list(getattr(obj, "segments", []) or [])

    summary = {
        "object_type": object_type,
        "object_module": type(obj).__module__,
        "name": make_json_safe(getattr(obj, "name", None)),
        "description": make_json_safe(getattr(obj, "description", None)),
        "file_origin": make_json_safe(getattr(obj, "file_origin", None)),
        "n_segments": int(len(segments)),
        "n_units": count_items(obj, "units"),
        "n_channel_indexes": count_items(obj, "channel_indexes"),
        "n_recordingchannelgroups": count_items(obj, "recordingchannelgroups"),
    }

    summary.update(annotation_summary(obj))

    segment_summaries = [summarize_segment(seg, include_object_details) for seg in segments]
    summary["segments"] = segment_summaries

    # Dataset-level totals for this file.
    summary["n_spiketrains"] = int(sum(seg.get("n_spiketrains", 0) for seg in segment_summaries))
    summary["n_spikes_total"] = int(sum(seg.get("n_spikes_total", 0) for seg in segment_summaries))
    summary["n_analogsignals"] = int(sum(seg.get("n_analogsignals", 0) for seg in segment_summaries))
    summary["n_events"] = int(sum(seg.get("n_events", 0) for seg in segment_summaries))
    summary["n_event_times_total"] = int(sum(seg.get("n_event_times_total", 0) for seg in segment_summaries))
    summary["n_epochs"] = int(sum(seg.get("n_epochs", 0) for seg in segment_summaries))
    summary["n_epoch_times_total"] = int(sum(seg.get("n_epoch_times_total", 0) for seg in segment_summaries))

    summary["has_spike_metadata"] = summary["n_spiketrains"] > 0
    summary["has_lfp_metadata"] = summary["n_analogsignals"] > 0
    summary["has_event_metadata"] = summary["n_events"] > 0
    summary["has_epoch_metadata"] = summary["n_epochs"] > 0

    return make_json_safe(summary)


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------

def load_pickle(file_path):
    install_old_neo_pickle_patches(verbose=False)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def extract_one_file(file_path, dataset_path, include_object_details=True):
    file_metadata = parse_touchandsee_file_metadata(file_path, dataset_path)

    result = {
        "file_metadata": file_metadata,
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
        obj = load_pickle(file_path)
        result["touchandsee_extraction"]["success"] = True
        result["touchandsee_extraction"]["object_type"] = type(obj).__name__
        result["touchandsee_extraction"]["object_summary"] = summarize_loaded_object(
            obj,
            include_object_details=include_object_details,
        )
    except Exception as error:
        result["touchandsee_extraction"]["success"] = False
        result["touchandsee_extraction"]["error"] = repr(error)
        result["touchandsee_extraction"]["traceback"] = traceback.format_exc()

    return make_json_safe(result)


def build_dataset_summary(file_results):
    # Be defensive: if a buggy previous conversion returned a string, skip it
    # instead of crashing while writing the JSON summary.
    file_results = [x for x in file_results if isinstance(x, dict)]

    successful = [
        x for x in file_results
        if x.get("touchandsee_extraction", {}).get("success")
    ]

    failed = [
        x for x in file_results
        if not x.get("touchandsee_extraction", {}).get("success")
    ]

    subjects = sorted(set(
        x.get("file_metadata", {}).get("subject_id")
        for x in file_results
        if x.get("file_metadata", {}).get("subject_id")
    ))

    sessions = sorted(set(
        x.get("file_metadata", {}).get("session_id")
        for x in file_results
        if x.get("file_metadata", {}).get("session_id")
    ))

    totals = {
        "n_spiketrains": 0,
        "n_spikes_total": 0,
        "n_units": 0,
        "n_analogsignals": 0,
        "n_events": 0,
        "n_event_times_total": 0,
        "n_epochs": 0,
        "n_epoch_times_total": 0,
        "n_sessions_with_spike_metadata": 0,
        "n_sessions_with_lfp_metadata": 0,
        "n_sessions_with_event_metadata": 0,
    }

    for item in successful:
        summary = item.get("touchandsee_extraction", {}).get("object_summary") or {}
        totals["n_spiketrains"] += summary.get("n_spiketrains") or 0
        totals["n_spikes_total"] += summary.get("n_spikes_total") or 0
        totals["n_units"] += summary.get("n_units") or 0
        totals["n_analogsignals"] += summary.get("n_analogsignals") or 0
        totals["n_events"] += summary.get("n_events") or 0
        totals["n_event_times_total"] += summary.get("n_event_times_total") or 0
        totals["n_epochs"] += summary.get("n_epochs") or 0
        totals["n_epoch_times_total"] += summary.get("n_epoch_times_total") or 0
        totals["n_sessions_with_spike_metadata"] += 1 if summary.get("has_spike_metadata") else 0
        totals["n_sessions_with_lfp_metadata"] += 1 if summary.get("has_lfp_metadata") else 0
        totals["n_sessions_with_event_metadata"] += 1 if summary.get("has_event_metadata") else 0

    return {
        "n_files": int(len(file_results)),
        "n_pickle_files": int(len(file_results)),
        "n_pickle_success": int(len(successful)),
        "n_pickle_failure": int(len(failed)),
        "n_subjects_detected": int(len(subjects)),
        "subjects_detected": subjects,
        "n_sessions_detected": int(len(sessions)),
        "sessions_detected": sessions,
        **{k: int(v) for k, v in totals.items()},
    }

def extract_touchandsee_dataset(
    dataset_path,
    output_folder=None,
    test_one=False,
    include_object_details=True,
):
    dataset_path = Path(dataset_path)

    if output_folder is None:
        output_folder = Path(os.getcwd()) / "outputs" / "extracted_metadata"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    pickle_files = find_pickle_files(dataset_path)

    if test_one:
        pickle_files = pickle_files[:1]

    print("Found pickle files:", len(pickle_files))

    results = []
    for i, file_path in enumerate(pickle_files, start=1):
        print(f"[{i}/{len(pickle_files)}] Extracting:", file_path.name)
        result = extract_one_file(
            file_path=file_path,
            dataset_path=dataset_path,
            include_object_details=include_object_details,
        )
        success = result.get("touchandsee_extraction", {}).get("success")
        if success:
            summary = result["touchandsee_extraction"].get("object_summary") or {}
            print(
                "  OK:",
                "segments=", summary.get("n_segments"),
                "spiketrains=", summary.get("n_spiketrains"),
                "spikes=", summary.get("n_spikes_total"),
                "analogsignals=", summary.get("n_analogsignals"),
                "events=", summary.get("n_events"),
            )
        else:
            print("  FAIL:", result.get("touchandsee_extraction", {}).get("error"))
        results.append(result)

    output = {
        "dataset_name": dataset_path.name,
        "dataset_folder": str(dataset_path),
        "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extractor": EXTRACTOR_NAME,
        "include_object_details": bool(include_object_details),
        "dataset_summary": build_dataset_summary(results),
        "files": results,
    }

    suffix = "_touchandsee_internal_metadata_TEST.json" if test_one else "_touchandsee_internal_metadata.json"
    output_path = output_folder / (dataset_path.name + suffix)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nFinished.")
    print("Output:", output_path)
    print("Dataset summary:")
    print(json.dumps(output["dataset_summary"], indent=2))

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract compact internal metadata from old TouchAndSee Neo pickle files."
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
        "--test_one",
        action="store_true",
        help="Extract only the first pickle file for debugging.",
    )
    parser.add_argument(
        "--no_object_details",
        action="store_true",
        help="Skip per-object previews and keep only compact counts.",
    )

    args = parser.parse_args()

    extract_touchandsee_dataset(
        dataset_path=args.dataset_path,
        output_folder=args.output_folder,
        test_one=args.test_one,
        include_object_details=not args.no_object_details,
    )
