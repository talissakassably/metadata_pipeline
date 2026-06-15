# -*- coding: utf-8 -*-
"""General helper functions for CA1 metadata meta-analysis."""

import math

def safe_get(mapping, key, default=None):
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return default


def as_bool(value):
    return bool(value) if value is not None else False


def to_number(value, default=0):
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return default
            return value
        return float(value)
    except Exception:
        return default


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def list_to_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(str(k) for k in value.keys())
    return str(value)


def infer_dataset_short_name(dataset_name):
    name = str(dataset_name)

    if "40faae41" in name:
        return "openfield_ca1"

    if "885b4936" in name:
        return "nwb_ca1"

    if "d406a98c" in name or "002061" in name:
        return "legacy_touchscreen"

    if "p25b4e" in name or "01681" in name or "Pennartz" in name:
        return "touchandsee"

    return name[:40]


def infer_source_type(dataset):
    extractor = str(dataset.get("extractor", "")).lower()
    dataset_name = str(dataset.get("dataset_name", "")).lower()

    if "sessions" in dataset and "openfield" in extractor:
        return "openfield"

    if "sessions" in dataset and "legacy_touchscreen" in extractor:
        return "legacy_touchscreen"

    # New TouchAndSee internal Neo pickle extractor output
    if "files" in dataset and ("touchandsee" in extractor or "01681" in dataset_name or "p25b4e" in dataset_name):
        return "regular_pickle_or_touchandsee"

    if "files" in dataset:
        extensions = dataset.get("extensions_searched", [])
        if ".nwb" in extensions:
            return "nwb"
        if ".pkl" in extensions:
            return "regular_pickle_or_touchandsee"

    return "unknown"


def has_any_text(value):
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return len(value) > 0
    return str(value).strip() != ""


def sum_dict_values(d):
    if not isinstance(d, dict):
        return 0
    total = 0
    for value in d.values():
        total += to_number(value, default=0)
    return total
